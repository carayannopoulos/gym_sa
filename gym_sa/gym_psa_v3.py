import multiprocessing as mp
import time
from typing import Callable, Optional, Tuple, Any, List, Dict
import numpy as np
import cloudpickle

from .annealer import Annealer
from .utils import softmax, set_seed
from .logger import CSVLogger
import sys
import os
import faulthandler
import copy
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

    n = 0
    # Enable faulthandler to capture hard crashes in worker processes
    with open(f"worker_{os.getpid()}.log", "w") as f:
        f.write(f"worker {os.getpid()} started\n")
    try:
        _fh_log = open(f"psa_fault_worker_{os.getpid()}.log", "w")
        faulthandler.enable(_fh_log)
    except Exception:
        # Best-effort; ignore if enabling fails
        pass
    annealer = annealer_constructor.var()

    while True:

        try:
            cmd, data = remote.recv()

            if cmd == "run_chain":
                try:
                    result = _run_chain(annealer, data["n_steps"], data["seed"])
                    remote.send(result)
                except Exception as e:
                    print(f"Error in run_chain: {e}")
                    import traceback

                    tb = traceback.format_exc()
                    print(tb)
                    # Send structured error back to parent
                    error_info = {
                        "error": True,
                        "cmd": "run_chain",
                        "exc_type": type(e).__name__,
                        "exc_msg": str(e),
                        "traceback": tb,
                    }
                    remote.send(("__EXC__", error_info))

            elif cmd == "reset":
                try:
                    annealer.reset()
                    remote.send(True)
                except Exception as e:
                    print(f"Error in reset: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send(False)

            elif cmd == "set_state":
                n += 1
                with open(f"worker_{os.getpid()}.log", "a") as f:
                    f.write(f"Setting state for remote {remote} {n}\n")
                try:
                    with open(f"worker_{os.getpid()}.log", "a") as f:
                        f.write(f"Setting state for remote {remote} {n}\n")
                    annealer.set_state(data["state"], data["objective"])
                    with open(f"worker_{os.getpid()}.log", "a") as f:
                        f.write(f"State set for remote {remote} {n}\n")
                    remote.send(True)
                    with open(f"worker_{os.getpid()}.log", "a") as f:
                        f.write(f"State sent for remote {remote} {n}\n")
                except Exception as e:
                    with open(f"worker_{os.getpid()}.log", "a") as f:
                        f.write(f"Error in set_state: {e} {n}\n")
                        # f.write(f"traceback: {traceback.format_exc()}\n")
                    print(f"Error in set_state: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send(False)

            elif cmd == "get_current_objective":
                try:
                    remote.send(annealer.current_objective)
                except Exception as e:
                    print(f"Error in get_current_objective: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send(None)

            elif cmd == "get_best_state":
                try:
                    best_state, best_objective = annealer.get_best_state()
                    remote.send((best_state, best_objective))
                except Exception as e:
                    print(f"Error in get_best_state: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send((None, None))

            elif cmd == "get_stats":
                try:
                    stats = annealer.get_stats()
                    remote.send(stats)
                except Exception as e:
                    print(f"Error in get_stats: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send({})

            elif cmd == "set_temperature":
                try:
                    annealer.temperature = data["temperature"]
                    remote.send(True)
                except Exception as e:
                    print(f"Error in set_temperature: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send(False)

            elif cmd == "get_temperature":
                try:
                    remote.send(annealer.temperature)
                except Exception as e:
                    print(f"Error in get_temperature: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send(None)

            elif cmd == "clear_accepted_energy":
                try:
                    annealer.accepted_energy.clear()
                    remote.send(True)
                except Exception as e:
                    print(f"Error in clear_accepted_energy: {e}")
                    import traceback

                    traceback.print_exc()
                    remote.send(False)

            elif cmd == "close":
                remote.close()
                break

            else:
                raise ValueError(f"Invalid command: {cmd}")
        except EOFError:
            with open(f"psa_fault_worker_{os.getpid()}.log", "a") as f:
                f.write(f"EOFError: {e}\n")
                f.write(f"traceback: {traceback.format_exc()}\n")
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

    # Run the chain for n_steps
    for _ in range(n_steps):
        annealer.step()

    # Get final stats
    best_state, best_objective = annealer.get_best_state()
    stats = annealer.get_stats()
    acceptance_rate = stats["mean_acceptance_rate"]

    return (
        annealer.accepted_energy,
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
        save_file: str = "save_dict.pkl",
        save_freq: int = None,
        save_progress: bool = False,
        load_file: str = None,
        load_progress: bool = False,
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
        self.save_file = save_file
        self.save_freq = save_freq
        self.save_progress = save_progress
        self.load_file = load_file
        self.load_progress = load_progress

        self.debug_file = "psa_debug.txt"
        with open(self.debug_file, "w") as f:
            f.write(f"PSA debug:\n")

        env_params_list = []
        for i in range(self.n_chains):
            e = copy.deepcopy(env_params)
            e["rank"] = i
            env_params_list.append(e)

        self.annealers_constructors = [
            CloudpickleWrapper(
                make_annealer(
                    self.env_constructor,
                    env_params_list[i],
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
        )

        # Track best state across all chains
        self.global_best_state = None
        self.global_best_objective = float("-inf")

        # Track start time for runtime monitoring
        self.start_time = None

        if self.load_progress:
            self.load_for_restart()
            # print("self.temperature: ", self.temperature)

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
        for remote in self.remotes:
            remote.send(("get_current_objective", None))

        return [remote.recv() for remote in self.remotes]

    def run_remote_chains(self, n_steps: int, seed: int = None):
        """
        Run the chains in the remote processes.
        """
        self.log_debug(f"running chains in parallel")
        for remote in self.remotes:
            remote.send(("run_chain", {"n_steps": n_steps, "seed": seed}))

        self.log_debug(f"waiting for chains to finish")
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                # Detect structured exception from worker and re-raise with full traceback
                if (
                    isinstance(result, tuple)
                    and len(result) == 2
                    and result[0] == "__EXC__"
                ):
                    error_info = result[1]
                    tb = error_info.get("traceback", "")
                    exc_type = error_info.get("exc_type", "Exception")
                    exc_msg = error_info.get("exc_msg", "")
                    print(f"Worker {i} raised {exc_type}: {exc_msg}")
                    if tb:
                        print(tb)
                    raise RuntimeError(
                        f"Worker {i} failed during {error_info.get('cmd')}: {exc_type}: {exc_msg}"
                    )
                results.append(result)
            except (BrokenPipeError, ConnectionResetError, EOFError) as e:
                print(f"Error receiving result from worker {i}: {e}")
                print(f"Worker {i} has crashed or disconnected")
                # Return None to indicate failure
                results.append((None, None, None, None, None, None))

        # Check if any workers failed
        failed_workers = [i for i, result in enumerate(results) if result[0] is None]
        if failed_workers:
            raise RuntimeError(f"Workers {failed_workers} failed or disconnected")

        self.log_debug(f"chains finished")
        return results

    def reset_remote_chains(self):
        """
        Reset the chains in the remote processes.
        """

        for remote in self.remotes:
            remote.send(("reset", None))

        return [remote.recv() for remote in self.remotes]

    def set_state_remote_chains(self, states: List[Any], objectives: List[float]):
        """
        Set the state of the chains in the remote processes.
        """
        outputs = []
        for remote, state, objective in zip(self.remotes, states, objectives):
            # alive = remote.poll(timeout=10)
            alive = not remote.closed
            if not alive:
                self.log_debug(
                    f"remote {remote} is not alive in set_state_remote_chains"
                )
                sys.exit()
            #     continue
            state_size = sys.getsizeof(pickle.dumps(state))
            self.log_debug(f"setting state for remote {remote} {state_size}")
            remote.send(("set_state", {"state": state, "objective": objective}))
            self.log_debug(f"waiting for state to be set for remote {remote}")
            outputs.append(remote.recv())
            self.log_debug(f"state set for remote {remote}")

        self.log_debug(f"states set for all remote chains")

        # for i, remote in enumerate(self.remotes):
        #     self.log_debug(f"waiting for state to be set for remote {remote} {i}")
        #     outputs.append(remote.recv())
        #     self.log_debug(f"state set for remote {remote} {i}: {outputs[i]}")

        return outputs

    def set_temperature_remote_chains(self, temperature: float):
        """
        Set the temperature of the chains in the remote processes.
        """
        for remote in self.remotes:
            # alive = remote.poll(timeout=10)
            alive = not remote.closed
            if not alive:
                self.log_debug(
                    f"remote {remote} is not alive in set_temperature_remote_chains"
                )
                sys.exit()
            remote.send(("set_temperature", {"temperature": temperature}))

        return [remote.recv() for remote in self.remotes]

    def get_temperature_remote_chains(self):
        """
        Get the temperature of the chains in the remote processes.
        """
        for remote in self.remotes:
            remote.send(("get_temperature", None))

        return [remote.recv() for remote in self.remotes]

    def clear_accepted_energy_remote_chains(self):

        for remote in self.remotes:
            remote.send(("clear_accepted_energy", None))

        return [remote.recv() for remote in self.remotes]

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

        n_set_states = 1

        try:
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
        finally:
            # Ensure proper cleanup of worker processes
            self.log_debug(f"cleaning up worker processes")
            self._cleanup_workers()
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

    def _cleanup_workers(self):
        """Clean up worker processes by sending close commands and joining them."""
        try:
            # Send close command to all workers
            for i, remote in enumerate(self.remotes):
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    print(f"Worker {i} connection already closed: {e}")
                    # Worker already closed
                    pass

            # Close all remote connections
            for remote in self.remotes:
                try:
                    remote.close()
                except:
                    pass

            # Join all processes with timeout
            for process in self.processes:
                try:
                    process.join(timeout=5)  # Wait up to 5 seconds
                    if process.is_alive():
                        print(f"Process {process.pid} still alive, terminating...")
                        process.terminate()
                        process.join(timeout=2)
                        if process.is_alive():
                            print(f"Process {process.pid} still alive, killing...")
                            process.kill()
                except Exception as e:
                    print(f"Error joining process {process.pid}: {e}")

            self.log_debug(f"worker cleanup completed")

        except Exception as e:
            print(f"Error during worker cleanup: {e}")
            # Force terminate all processes
            for process in self.processes:
                try:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1)
                        if process.is_alive():
                            process.kill()
                except:
                    pass

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
