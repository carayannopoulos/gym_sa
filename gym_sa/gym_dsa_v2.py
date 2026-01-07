import multiprocessing as mp
import time
from typing import Callable, Optional, Tuple, Any, List, Dict
import numpy as np
import cloudpickle
from .job_manager import JobManager
from .annealer import Annealer
from .utils import softmax, set_seed
from .logger import CSVLogger
import sys
import os
import faulthandler
import copy
import pickle
import traceback
import time
import os
import fcntl
import termios
import array

from .sa_rolloutworker import SA_RolloutWorker
import ray


class DistributedAnnealer_batch:
    def __init__(
        self,
        runtime_env: Dict,
        env_constructor: Callable,
        env_params: Dict,
        n_chains: int,
        job_manager_kwargs: Dict,
        ray_address: str = "auto",
        temperature_schedule: str = "geometric",
        temp_params: Dict = None,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.99,
        min_temperature: float = 1e-3,
        mixing_frequency: int = 100,
        max_steps: int = 1000,
        min_acceptance_rate: float = 0.01,
        max_runtime: float = 3600,  # 1 hour default
        base_seed: Optional[int] = None,
        verbose: bool = False,
        log_file: str = "parallel_annealing_log.csv",
        save_file: str = "save_dict.pkl",
        save_freq: int = None,
        save_progress: bool = False,
        load_file: str = None,
        load_progress: bool = False,
        restart: bool = False,
    ):
        """
        Initialize parallel simulated annealing with multiple chains.

        Args:
            env_constructor: Function to create environment instances
            n_chains: Number of parallel chains to run
            initial_temperature: Starting temperature for annealing
            cooling_rate: Multiplicative factor for temperature decay
            min_temperature: Minimum temperature to stop cooling
            mixing_frequency: How often to mix states between chains
            max_steps: Maximum number of steps to run
            min_acceptance_rate: Minimum acceptance rate before stopping
            max_runtime: Maximum runtime in seconds
            base_seed: Base seed for random number generation
            log_file: Path to CSV file for logging metrics
        """
        self.runtime_env = runtime_env
        self.ray_address = ray_address
        self.env_constructor = env_constructor
        self.n_chains = n_chains
        self.temperature_schedule = temperature_schedule
        self.temp_params = temp_params
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.mixing_frequency = mixing_frequency
        self.max_steps = max_steps
        self.min_acceptance_rate = min_acceptance_rate
        self.max_runtime = max_runtime
        self.verbose = verbose
        self.save_file = save_file
        self.save_freq = save_freq
        self.save_progress = save_progress
        self.load_file = load_file
        self.load_progress = load_progress
        self.job_manager_kwargs = job_manager_kwargs
        self.debug_file = "dsa_debug.txt"
        with open(self.debug_file, "w") as f:
            f.write(f"DSA debug:\n")
        self.restart = restart
        self.env_params_list = []
        for i in range(self.n_chains):
            e = copy.deepcopy(env_params)
            e["rank"] = i
            self.env_params_list.append(e)

        self.init_ray()

        # Initialize random number generator
        self.rng = np.random.default_rng(base_seed)

        # if adaptive cooling, initialize temperature
        # from D. Kropaczek, “COPERNICUS: A multi-cycle optimization code for nuclear fuel based on parallel simulated annealing with mixing of states,” Progress in Nuclear Energy.
        if self.temperature_schedule == "adaptive":
            initial_energy = self.get_current_objective_remote_chains()
            # print(f"initial_energy: {initial_energy}")
            self.initial_temperature = self.temp_params["alpha"] * np.std(
                initial_energy
            )
            self.initial_temperature = max(
                self.initial_temperature, self.temp_params["min_initial_temp"]
            )
            self.temperature = self.initial_temperature

        else:
            self.temperature = self.initial_temperature

        # Initialize logger
        self.logger = CSVLogger(
            log_file,
            fieldnames=[
                "step",
                "best_objective",
                "avg_acceptance_rate",
                "temperature",
                "runtime",
            ],
            restart=self.restart,
        )

        # Track best state across all chains
        self.global_best_state = None
        self.global_best_objective = float("-inf")

        # Track start time for runtime monitoring
        self.start_time = None

        if self.load_progress:
            self.load_for_restart()
            # print("self.temperature: ", self.temperature)

    def init_ray(self):
        """
        Initialize Ray and spawn rollout workers.
        """
        print(f"Initializing Ray at address={self.ray_address} ...")
        ray.init(
            # address=self.ray_address,
            local_mode=True,
            ignore_reinit_error=True,
            runtime_env=self.runtime_env,
        )

        self.job_manager = JobManager.remote(**self.job_manager_kwargs)
        for e in self.env_params_list:
            e["job_manager"] = self.job_manager

        print(f"Spawning {self.n_chains} rollout workers ...")
        self.workers = [
            SA_RolloutWorker.remote(
                self.env_constructor,
                env_params=self.env_params_list[i],
                temperature_schedule=self.temperature_schedule,
                initial_temperature=self.initial_temperature,
                cooling_rate=self.cooling_rate,
                min_temperature=self.min_temperature,
                verbose=False,
            )
            for i in range(self.n_chains)
        ]

        ray.get([w.ping.remote() for w in self.workers])

        self.reset_remote_chains()

    def __del__(self):
        """Destructor to ensure cleanup happens even if object is garbage collected."""
        try:
            self._cleanup_workers()
        except:
            pass  # Ignore errors during cleanup in destructor

    def log_debug(self, message):
        with open(self.debug_file, "a") as f:
            f.write(f"{message}\n")

    def get_current_objective_remote_chains(self):
        """
        Get the current objective of the chains in the remote processes.
        """
        obj_refs = [worker.get_current_objective.remote() for worker in self.workers]

        return ray.get(obj_refs)

    def run_remote_chains(self, n_steps: int, seed: int = None):
        """
        Run the chains in the remote processes.
        """
        self.log_debug(f"running chains in parallel")
        refs = []
        for worker in self.workers:
            refs.append(worker.run_chain.remote(n_steps, seed))
        self.log_debug(f"waiting for chains to finish")
        results = ray.get(refs)
        return results

    def reset_remote_chains(self):
        """
        Reset the chains in the remote processes.
        """

        ray.get([worker.reset.remote() for worker in self.workers])

        return True

    def set_state_remote_chains(self, states: List[Any], objectives: List[float]):
        """
        Set the state of the chains in the remote processes.
        """

        ray.get(
            [
                worker.set_state.remote(state, objective)
                for worker, state, objective in zip(self.workers, states, objectives)
            ]
        )

        return True

    def set_temperature_remote_chains(self, temperature: float):
        """
        Set the temperature of the chains in the remote processes.
        """
        ray.get([worker.set_temperature.remote(temperature) for worker in self.workers])

        return True

    def get_temperature_remote_chains(self):
        """
        Get the temperature of the chains in the remote processes.
        """
        return ray.get([worker.get_temperature.remote() for worker in self.workers])

    def clear_accepted_energy_remote_chains(self):

        ray.get([worker.clear_accepted_energy.remote() for worker in self.workers])

        return True

    def save_for_restart(self, states, objectives, temperature):
        """
        Save the states, objectives, and temperature for restart.
        """
        save_dict = {
            "states": states,
            "objectives": objectives,
            "temperature": temperature,
        }
        with open(self.save_file, "wb") as f:
            pickle.dump(save_dict, f)

        return save_dict

    def load_for_restart(self):
        """
        Load the states, objectives, and temperature for restart.
        """
        with open(self.load_file, "rb") as f:
            save_dict = pickle.load(f)

        self.temperature = save_dict["temperature"]

        self.set_state_remote_chains(save_dict["states"], save_dict["objectives"])

    def _mix_states(
        self, best_states: List[Any], best_objectives: List[float]
    ) -> List[Any]:
        """
        Mix states between chains using softmax sampling.

        Args:
            chain_states: List of (state, objective) tuples from each chain

        Returns:
            List of new states for each chain
        """
        self.log_debug(f"mixing states")
        # Extract objectives for softmax sampling
        objectives = np.array(best_objectives)
        self.log_debug(f"objectives: {objectives}")

        # Calculate softmax probabilities
        probs = softmax(objectives)
        self.log_debug(f"probs: {probs}")
        # Sample new states for each chain
        new_states = []
        new_objectives = []
        for _ in range(self.n_chains):
            # Sample a state based on softmax probabilities
            idx = self.rng.choice(len(best_states), p=probs)
            new_states.append(best_states[idx])
            new_objectives.append(best_objectives[idx])
        self.log_debug(f"new states: {new_states}")
        self.log_debug(f"new objectives: {new_objectives}")
        return new_states, new_objectives

    def run(self) -> Tuple[Any, float]:
        """
        Run the parallel annealing process until termination.

        Returns:
            Tuple of (best_state, best_objective) found across all chains
        """
        self.start_time = time.time()
        self.log_debug(f"beginning of run")
        # compute the number of outer iterations
        n_outer_iterations = self.max_steps // self.mixing_frequency + 1

        n_set_states = 0

        for step in range(n_outer_iterations):
            # Run chains in parallel
            self.log_debug(f"running chains in parallel")
            results = self.run_remote_chains(self.mixing_frequency)

            # get the output of the annealers
            self.accepted_energies = [r[0] for r in results]
            # print(f"accepted_energies: {self.accepted_energies}")
            best_states = [r[1] for r in results]
            best_objectives = [r[2] for r in results]
            acceptance_rates = [r[3] for r in results]
            current_states = [r[4] for r in results]
            current_objectives = [r[5] for r in results]
            self.log_debug(f"output of chains obtained")
            # Update global best
            for state, obj in zip(best_states, best_objectives):
                if obj > self.global_best_objective:
                    self.global_best_state = state
                    self.global_best_objective = obj
            self.log_debug(f"global best updated")
            # Log metrics
            self._log_metrics(step, best_states, best_objectives, acceptance_rates)
            self.log_debug(f"metrics logged")
            # Check termination conditions
            if self._should_terminate(step, acceptance_rates):
                break
            self.log_debug(f"termination conditions checked")
            # mix the states
            try:
                self.log_debug(f"about to mix states")
                new_states, new_objectives = self._mix_states(
                    best_states, best_objectives
                )
                self.log_debug(f"states mixed")
            except:
                self.log_debug(f"error in mix_states")
            # Update chain states for next iteration
            self.log_debug(f"about to set states {n_set_states}")
            self.set_state_remote_chains(new_states, new_objectives)
            n_set_states += 1
            self.log_debug(f"states updated {n_set_states}")
            # update the temperature if adaptive
            if self.temperature_schedule == "adaptive":
                s = 1 / self.temperature
                rho = np.mean(acceptance_rates)
                accepted_energies = np.concatenate(self.accepted_energies)
                self.log_debug(f"accepted energies concatenated")

                # make sure there are some accepted energies
                if len(accepted_energies) > 0:
                    sig = np.std(accepted_energies)
                    self.log_debug(f"standard deviation calculated")
                    # make sure the standard deviation isn't zero
                    if sig > 1e-8:
                        G = (4 * rho * (1 - rho) ** 2) / (2 - rho) ** 2
                        s_new = (
                            s
                            + self.temp_params["lambda"]
                            * (1 / sig)
                            * (1 / (s**2 * sig**2))
                            * G
                        )
                        self.temperature = max(1 / s_new, self.min_temperature)
                        self.log_debug(f"temperature updated")

                # clear the circular buffers (more efficient)
                # self.clear_accepted_energy_remote_chains()
                self.log_debug(f"accepted energies cleared")
                # set the temperature for each annealer
                self.set_temperature_remote_chains(self.temperature)
                self.log_debug(f"temperature set for each annealer")
            elif self.temperature_schedule == "geometric":
                self.temperature = self.get_temperature_remote_chains()[0]
                self.log_debug(f"temperature set for each annealer")
            # Summarize step
            if self.verbose:
                print(f"Step {step+1} / {n_outer_iterations}")
                print(f"Best objective: {self.global_best_objective}")
                print(f"Avg acceptance rate: {np.mean(acceptance_rates)}")
                if self.temperature_schedule == "adaptive":
                    print(f"Temperature: {self.temperature}")
                elif self.temperature_schedule == "geometric":
                    print(
                        f"Temperature: {self.initial_temperature * (self.cooling_rate ** (step * self.mixing_frequency))}"
                    )
                print(f"Runtime: {time.time() - self.start_time}")
            self.log_debug(f"step summarized")
            if self.save_progress and step % self.save_freq == 0:
                self.save_for_restart(
                    current_states, current_objectives, self.temperature
                )
                self.log_debug(f"save for restart")

        self.log_debug(f"run completed")
        return self.global_best_state, self.global_best_objective

    def _should_terminate(self, step: int, acceptance_rates: List[float]) -> bool:
        """Check if we should terminate the annealing process."""

        # Check acceptance rate
        avg_acceptance = np.mean(acceptance_rates)
        if avg_acceptance < self.min_acceptance_rate:
            return True

        # Check runtime
        if time.time() - self.start_time > self.max_runtime:
            return True

        return False

    def _log_metrics(
        self,
        step: int,
        best_states: List[Any],
        best_objectives: List[float],
        acceptance_rates: List[float],
    ) -> None:
        """Log global metrics to CSV file."""
        metrics = {
            "step": step,
            "best_objective": np.round(self.global_best_objective, 2),
            "avg_acceptance_rate": np.round(np.mean(acceptance_rates), 2),
            "temperature": np.round(self.temperature, 2),
            "runtime": np.round(time.time() - self.start_time, 2),
        }
        self.logger.log(metrics)
