from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, List, Optional

import ray

logger = logging.getLogger(__name__)


@dataclass
class SimulationRequest:
    """Metadata describing a single simulation invocation."""

    request_id: str
    payload: Any
    retries: int = 0
    future: Future = field(default_factory=Future, repr=False)


@dataclass
class BatchContext:
    """Bookkeeping for a submitted batch."""

    batch_id: str
    requests: List[SimulationRequest]
    manifest_path: Path
    script_path: Path
    log_path: Path
    job_name: str
    num_nodes: int
    job_id: Optional[str] = None
    submitted_at: Optional[float] = None


@ray.remote(num_cpus=0.1)
class JobManager:
    """Coordinate simulation batching and Slurm submissions for distributed RL."""

    def __init__(
        self,
        batch_size: int,
        batch_runner_command: List[str],
        slurm_template_path: str,
        staging_dir: str,
        partition: str = "sched_mit_nse",
        num_nodes: int = 1,
        poll_interval: float = 5.0,
        max_task_retries: int = 2,
    ) -> None:
        self.batch_size = batch_size
        self.batch_runner_command = list(batch_runner_command)
        self.slurm_template_path = Path(slurm_template_path)
        self.staging_dir = Path(staging_dir)
        self.num_nodes = num_nodes
        self.poll_interval = poll_interval
        self.max_task_retries = max_task_retries

        self._pending: Deque[SimulationRequest] = deque()
        self._pending_condition = threading.Condition()
        self._active_batch: Optional[BatchContext] = None
        self._running = True

        self._staging_dir = self._ensure_directory(self.staging_dir)
        self._submission_thread = threading.Thread(
            target=self._batch_worker_loop, name="JobManagerBatchWorker", daemon=True
        )
        self._submission_thread.start()

        logger.info(
            "JobManager initialized: batch_size=%s, staging_dir=%s",
            self.batch_size,
            self._staging_dir,
        )

    def submit_job(self, payload: Any) -> bool:
        """Schedule a simulation and block until it completes.

        Args:
            payload: Opaque data required by the batch runner to launch the simulation.

        Returns:
            True when the simulation finishes successfully.
        """
        request = SimulationRequest(request_id=str(uuid.uuid4()), payload=payload)
        with self._pending_condition:
            self._pending.append(request)
            self._pending_condition.notify()

        logger.debug("Enqueued simulation %s", request.request_id)
        return request.future.result()

    # ------------------------------------------------------------------ #
    # Internal helpers (executed on the head node worker thread)
    # ------------------------------------------------------------------ #

    def _batch_worker_loop(self) -> None:
        """Continuously build batches, submit them, and monitor completion."""
        while self._running:
            batch = self._collect_next_batch()
            if not batch:
                continue

            try:
                logger.info(
                    "Submitting batch %s with %d simulations",
                    batch.batch_id,
                    len(batch.requests),
                )
                job_id = self._submit_batch(batch)
                batch.job_id = job_id
                batch.submitted_at = time.time()
                self._active_batch = batch
                self._monitor_batch(batch)
            except Exception as exc:  # pragma: no cover - error path
                logger.exception("Batch %s failed before submission", batch.batch_id)
                self._handle_batch_exception(batch, exc)
            finally:
                self._active_batch = None

    def _collect_next_batch(self) -> Optional[BatchContext]:
        """Wait for enough pending requests to form a batch."""
        with self._pending_condition:
            while self._running and len(self._pending) < self.batch_size:
                self._pending_condition.wait()
            if not self._running:
                return None

            requests = [self._pending.popleft() for _ in range(self.batch_size)]

        batch_id = str(uuid.uuid4())
        batch_dir = self._staging_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = batch_dir / "batch_manifest.json"
        script_path = batch_dir / "submit.sh"
        log_path = batch_dir / "slurm.log"

        batch = BatchContext(
            batch_id=batch_id,
            requests=requests,
            manifest_path=manifest_path,
            script_path=script_path,
            log_path=log_path,
            job_name=f"sim-batch-{batch_id[:8]}",
            num_nodes=self.num_nodes,
        )

        self._write_manifest(batch)
        self._render_slurm_script(batch)

        return batch

    def _submit_batch(self, batch: BatchContext) -> str:
        """Invoke sbatch for the rendered script and return the Slurm job id."""
        import subprocess  # local import to keep module importable without subprocess

        cmd = ["sbatch", str(batch.script_path)]
        logger.debug("Running %s", " ".join(cmd))
        completed = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        )
        stdout = completed.stdout.strip()
        logger.debug("sbatch output: %s", stdout)
        job_id = self._parse_job_id(stdout)
        if not job_id:
            raise RuntimeError(f"Failed to parse job id from sbatch output: {stdout}")
        return job_id

    def _monitor_batch(self, batch: BatchContext) -> None:
        """Poll Slurm until the batch job completes."""
        while True:
            if not self._is_job_active(batch.job_id):
                break
            time.sleep(self.poll_interval)

        self._finalize_batch(batch)

    def _finalize_batch(self, batch: BatchContext) -> None:
        """Resolve futures based on the batch summary file."""
        summary_path = batch.manifest_path.with_name("batch_summary.json")
        if not summary_path.exists():
            logger.warning(
                "Summary file missing for batch %s; assuming success", batch.batch_id
            )
            for request in batch.requests:
                request.future.set_result(True)
            return

        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            logger.exception(
                "Malformed summary for batch %s; requeuing all tasks", batch.batch_id
            )
            self._requeue_requests(batch.requests)
            return

        failed: List[SimulationRequest] = []
        for request in batch.requests:
            result = summary.get(request.request_id, {})
            if result.get("status") == "success":
                request.future.set_result(True)
            else:
                failed.append(request)

        if failed:
            logger.info(
                "Batch %s completed with %d failures; retrying",
                batch.batch_id,
                len(failed),
            )
            self._requeue_requests(failed)

    def _requeue_requests(self, requests: List[SimulationRequest]) -> None:
        """Requeue failed simulations or mark them failed if retries exhausted."""
        with self._pending_condition:
            for request in requests:
                request.retries += 1
                if request.retries > self.max_task_retries:
                    request.future.set_exception(
                        RuntimeError(
                            f"Simulation {request.request_id} exceeded retry limit"
                        )
                    )
                    continue
                self._pending.appendleft(request)
            self._pending_condition.notify_all()

    def _handle_batch_exception(self, batch: BatchContext, exc: Exception) -> None:
        """Handle exceptions encountered during submission or monitoring."""
        logger.error("Batch %s failed: %s", batch.batch_id, exc)
        with self._pending_condition:
            for request in batch.requests:
                if request.future.done():
                    continue
                request.retries += 1
                if request.retries > self.max_task_retries:
                    request.future.set_exception(exc)
                else:
                    self._pending.appendleft(request)
            self._pending_condition.notify_all()

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _ensure_directory(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_manifest(self, batch: BatchContext) -> None:
        """Persist the batch specification for consumption by the runner."""
        manifest = {
            "batch_id": batch.batch_id,
            "requests": [
                {"request_id": request.request_id, "payload": request.payload}
                for request in batch.requests
            ],
        }
        batch.manifest_path.write_text(json.dumps(manifest, indent=2))

    def _render_slurm_script(self, batch: BatchContext) -> None:
        """Fill in the Slurm template for the current batch."""
        template = self.slurm_template_path.read_text()
        command = self._build_batch_command(batch)
        script = template.format(
            job_name=batch.job_name,
            num_nodes=batch.num_nodes,
            command=command,
            log_path=batch.log_path,
            partition=partition,
        )
        batch.script_path.write_text(script)

    def _build_batch_command(self, batch: BatchContext) -> str:
        """Return the command line that the Slurm allocation should execute."""
        cmd_parts = list(self.batch_runner_command)
        cmd_parts.extend(["--manifest", str(batch.manifest_path)])
        return " ".join(cmd_parts)

    @staticmethod
    def _parse_job_id(sbatch_output: str) -> Optional[str]:
        """Extract the job id from sbatch stdout."""
        tokens = sbatch_output.strip().split()
        if not tokens:
            return None
        for token in reversed(tokens):
            if token.isdigit():
                return token
        return None

    def _is_job_active(self, job_id: Optional[str]) -> bool:
        """Check whether the given Slurm job is still running or pending."""
        if not job_id:
            return False

        import subprocess  # local import

        cmd = ["squeue", "--noheader", "--job", job_id]
        completed = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        return completed.returncode == 0 and bool(completed.stdout.strip())

    def shutdown(self) -> None:
        """Stop the background worker thread and resolve outstanding futures."""
        self._running = False
        with self._pending_condition:
            self._pending_condition.notify_all()
        self._submission_thread.join(timeout=5)

        while self._pending:
            request = self._pending.popleft()
            if not request.future.done():
                request.future.set_exception(RuntimeError("Manager shutdown"))

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            if self._running:
                self.shutdown()
        except Exception:
            pass
