import multiprocessing as mp
import time
from typing import Callable, Optional, Tuple, Any, List, Dict
import numpy as np
import cloudpickle

from .annealer import Annealer
from .utils import softmax, set_seed
from .logger import CSVLogger
import sys

import pickle


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


def _worker(
    remote: Any,
    parent: Any,
    annealer_constructor: CloudpickleWrapper,
):
    annealer = annealer_constructor.var()

    while True:

        try:
            cmd, data = remote.recv()

            if cmd == "run_chain":
                result = _run_chain(annealer, data["n_steps"], data["seed"])
                remote.send(result)

            if cmd == "reset":
                annealer.reset()
                print(f"post reset current objective: {annealer.current_objective}")
                remote.send(True)

            if cmd == "set_state":
                annealer.set_state(data["state"])
                remote.send(True)

            if cmd == "get_current_objective":
                print(f"getting current objective")
                print(f"annealer.current_objective: {annealer.current_objective}")
                remote.send(annealer.current_objective)

            if cmd == "get_best_state":
                best_state, best_objective = annealer.get_best_state()
                remote.send((best_state, best_objective))

            if cmd == "get_stats":
                stats = annealer.get_stats()
                remote.send(stats)

            elif cmd == "close":
                remote.close()
                break
        except EOFError:
            # remote.close()
            break


def _run_chain(
    annealer: Annealer, n_steps: int, seed: int = None
) -> Tuple[Any, float, float, Any]:
    """
    Run a single annealing chain in a separate process.

    Args:
        chain_id: Identifier for this chain
        seed: Random seed for this chain
        initial_state: Optional initial state to start from

    Returns:
        Tuple of (best_state, best_objective, acceptance_rate, current_state)
    """
    # Set seed for this process
    if seed is not None:
        set_seed(seed)

    # annealer = Annealer(obj_dict=annealer_dict)

    # Run the chain for n_steps
    for _ in range(n_steps):
        annealer.step()

    # Get final stats
    best_state, best_objective = annealer.get_best_state()
    stats = annealer.get_stats()
    acceptance_rate = stats["mean_acceptance_rate"]

    return (
        annealer,
        best_state,
        best_objective,
        acceptance_rate,
        annealer.state,
        annealer.current_objective,
    )


class ParallelAnnealer:
    def __init__(
        self,
        env_constructor: Callable,
        env_params: Dict,
        n_chains: int,
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

        # initialize the annealers
        # self.annealers = [
        #     CloudpickleWrapper(
        #         Annealer(
        #             env_constructor=self.env_constructor,
        #             env_params=env_params,
        #             temperature_schedule=self.temperature_schedule,
        #             initial_temperature=self.initial_temperature,
        #             cooling_rate=self.cooling_rate,
        #             min_temperature=self.min_temperature,
        #             verbose=False,
        #         )
        #     )
        #     for i in range(self.n_chains)
        # ]

        self.annealers_constructors = [
            CloudpickleWrapper(
                make_annealer(
                    self.env_constructor,
                    env_params,
                    self.temperature_schedule,
                    self.initial_temperature,
                    self.cooling_rate,
                    self.min_temperature,
                    i,
                    verbose=False,
                    # self.verbose,
                )
            )
            for i in range(self.n_chains)
        ]

        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        # initialize the processes
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.n_chains)]
        )

        self.processes = []

        for work_remote, remote, annealer_constructor in zip(
            self.work_remotes, self.remotes, self.annealers_constructors
        ):
            process = ctx.Process(
                target=_worker, args=(work_remote, remote, annealer_constructor)
            )
            process.start()
            self.processes.append(process)
            work_remote.close()

        # reset all of the annealers
        self.reset_remote_chains()

        # Initialize random number generator
        self.rng = np.random.default_rng(base_seed)

        # if adaptive cooling, initialize temperature
        # from D. Kropaczek, “COPERNICUS: A multi-cycle optimization code for nuclear fuel based on parallel simulated annealing with mixing of states,” Progress in Nuclear Energy.
        if self.temperature_schedule == "adaptive":
            initial_energy = self.get_current_objective_remote_chains()
            print(f"initial_energy: {initial_energy}")
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
        )

        # Track best state across all chains
        self.global_best_state = None
        self.global_best_objective = float("-inf")

        # Track start time for runtime monitoring
        self.start_time = None

    def get_current_objective_remote_chains(self):
        """
        Get the current objective of the chains in the remote processes.
        """
        for remote in self.remotes:
            remote.send(("get_current_objective", None))

        return [remote.recv() for remote in self.remotes]

    def run_remote_chains(self, n_steps: int, seed: int = None):
        """
        Run the chains in the remote processes.
        """
        for remote in self.remotes:
            remote.send(("run_chain", {"n_steps": n_steps, "seed": seed}))

        results = [remote.recv() for remote in self.remotes]

        return results

    def reset_remote_chains(self):
        """
        Reset the chains in the remote processes.
        """
        for remote in self.remotes:
            remote.send(("reset", None))

        return [remote.recv() for remote in self.remotes]

    def set_state_remote_chains(self, states: List[Any]):
        """
        Set the state of the chains in the remote processes.
        """
        for remote, state in zip(self.remotes, states):
            remote.send(("set_state", {"state": state}))

        return [remote.recv() for remote in self.remotes]

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
        # Extract objectives for softmax sampling
        objectives = np.array(best_objectives)

        # Calculate softmax probabilities
        probs = softmax(objectives)

        # Sample new states for each chain
        new_states = []
        for _ in range(self.n_chains):
            # Sample a state based on softmax probabilities
            idx = self.rng.choice(len(best_states), p=probs)
            new_states.append(best_states[idx])

        return new_states

    def run(self) -> Tuple[Any, float]:
        """
        Run the parallel annealing process until termination.

        Returns:
            Tuple of (best_state, best_objective) found across all chains
        """
        self.start_time = time.time()

        # compute the number of outer iterations
        n_outer_iterations = self.max_steps // self.mixing_frequency + 1

        for step in range(n_outer_iterations):
            # Run chains in parallel

            results = self.run_remote_chains(self.mixing_frequency)

            # get the output of the annealers
            self.annealers = [r[0] for r in results]
            best_states = [r[1] for r in results]
            best_objectives = [r[2] for r in results]
            acceptance_rates = [r[3] for r in results]
            current_states = [r[4] for r in results]
            current_objectives = [r[5] for r in results]

            # Update global best
            for state, obj in zip(best_states, best_objectives):
                if obj > self.global_best_objective:
                    self.global_best_state = state
                    self.global_best_objective = obj

            # Log metrics
            self._log_metrics(step, best_states, best_objectives, acceptance_rates)

            # Check termination conditions
            if self._should_terminate(step, acceptance_rates):
                break

            # mix the states
            new_states = self._mix_states(best_states, best_objectives)
            # Update chain states for next iteration
            self.set_state_remote_chains(new_states)

            # update the temperature if adaptive
            if self.temperature_schedule == "adaptive":
                s = 1 / self.temperature
                rho = np.mean(acceptance_rates)
                # Old: list comprehension on growing lists
                # accepted_energies = [
                #     i for a in self.annealers for i in a.accepted_energy
                # ]

                # New: list comprehension on fixed-size circular buffers
                accepted_energies = [
                    i for a in self.annealers for i in a.accepted_energy
                ]
                # make sure there are some accepted energies
                if len(accepted_energies) > 0:
                    sig = np.std(accepted_energies)

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

                # print(f"sig: {sig}")
                # print(f"s: {s}")
                # print(f"s_new: {s_new}")
                # print(f"temperature: {self.temperature}")

                # Old: reset by creating new empty lists
                # for a in self.annealers:
                #     a.accepted_energy = []

                # New: clear the circular buffers (more efficient)
                for a in self.annealers:
                    a.accepted_energy.clear()

                # set the temperature for each annealer
                for a in self.annealers:
                    a.temperature = self.temperature

            elif self.temperature_schedule == "geometric":
                self.temperature = self.annealers[0].temperature

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


def is_pickleable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


def make_annealer(
    env_constructor,
    env_params,
    temperature_schedule,
    initial_temperature,
    cooling_rate,
    min_temperature,
    seed,
    verbose,
):
    """ """

    def _init():
        env = Annealer(
            env_constructor=env_constructor,
            env_params=env_params,
            temperature_schedule=temperature_schedule,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            seed=seed,
            verbose=verbose,
        )
        return env

    return _init
