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
import traceback
import time
import os
import fcntl
import termios
import array

from gym_sa import annealer


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
    receive_queue,  #: mp.connection.Connection,
    send_queue,  #: mp.connection.Connection,
    annealer_constructor: CloudpickleWrapper,
) -> None:
    # Import here to avoid a circular import

    # with open(f"worker_{os.getpid()}.log", "w") as f:
    #     f.write(f"worker {os.getpid()} started\n")

    annealer = annealer_constructor.var()
    reset_info: Optional[dict[str, Any]] = {}

    n = 0

    while True:
        try:
            # with open(f"worker_{os.getpid()}.log", "a") as f:
            #     f.write(f"=== WAITING FOR COMMAND ===\n")
            #     f.flush()

            cmd, data = receive_queue.get()

            # with open(f"worker_{os.getpid()}.log", "a") as f:
            #     f.write(f"=== RECEIVED: {cmd} ===\n")
            #     f.flush()

            if cmd == "run_chain":
                try:
                    result = _run_chain(annealer, data["n_steps"], data["seed"])
                    # with open(f"worker_{os.getpid()}.log", "a") as f:
                    #     for r in result:
                    #         f.write(f"{type(r)}--")
                    #     f.write(f"\n")
                    send_queue.put(result)

                except:
                    # with open(f"worker_{os.getpid()}.log", "a") as f:
                    #     f.write(f"error in run_chain\n")
                    #     traceback.print_exc(file=f)
                    send_queue.put(None)

            elif cmd == "reset":
                try:
                    annealer.reset()
                    send_queue.put(True)
                except:
                    # with open(f"worker_{os.getpid()}.log", "a") as f:
                    #     f.write(f"error in reset\n")
                    #     traceback.print_exc(file=f)
                    send_queue.put(False)

            elif cmd == "set_state":
                n += 1
                # with open(f"worker_{os.getpid()}.log", "a") as f:
                #     f.write(f"setting state for remote {n}\n")
                try:
                    # with open(f"worker_{os.getpid()}.log", "a") as f:
                    #     f.write(f"setting state for remote {remote} {n}\n")
                    annealer.set_state(data["state"], data["objective"])
                    # with open(f"worker_{os.getpid()}.log", "a") as f:
                    #     f.write(f"state set for remote {remote} {n}\n")
                    send_queue.put(True)
                except:
                    # with open(f"worker_{os.getpid()}.log", "a") as f:
                    #     f.write(f"error in set_state\n")
                    #     traceback.print_exc(file=f)
                    send_queue.put(False)

            elif cmd == "get_current_objective":
                try:
                    send_queue.put(annealer.current_objective)
                except Exception as e:
                    print(f"Error in get_current_objective: {e}")
                    import traceback

                    traceback.print_exc()
                    send_queue.put(None)

            elif cmd == "get_best_state":
                try:
                    best_state, best_objective = annealer.get_best_state()
                    send_queue.put((best_state, best_objective))
                except Exception as e:
                    print(f"Error in get_best_state: {e}")
                    import traceback

                    traceback.print_exc()
                    send_queue.put((None, None))

            elif cmd == "get_stats":
                try:
                    stats = annealer.get_stats()
                    send_queue.put(stats)
                except Exception as e:
                    print(f"Error in get_stats: {e}")
                    import traceback

                    traceback.print_exc()
                    send_queue.put({})

            elif cmd == "set_temperature":
                try:
                    annealer.temperature = data["temperature"]
                    send_queue.put(True)
                except Exception as e:
                    print(f"Error in set_temperature: {e}")
                    import traceback

                    traceback.print_exc()
                    send_queue.put(False)

            elif cmd == "get_temperature":
                try:
                    send_queue.put(annealer.temperature)
                except Exception as e:
                    print(f"Error in get_temperature: {e}")
                    import traceback

                    traceback.print_exc()
                    send_queue.put(None)

            elif cmd == "clear_accepted_energy":
                try:
                    annealer.accepted_energy.clear()
                    send_queue.put(True)
                except Exception as e:
                    print(f"Error in clear_accepted_energy: {e}")
                    import traceback

                    traceback.print_exc()
                    send_queue.put(False)

            elif cmd == "close":
                send_queue.close()
                break

        except EOFError:
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

    best_objective = annealer.current_objective
    best_state = copy.deepcopy(annealer.state)

    # Run the chain for n_steps
    for _ in range(n_steps):
        annealer.step()
        if annealer.current_objective > best_objective:
            best_objective = annealer.current_objective
            best_state = copy.deepcopy(annealer.state)

    # Get final stats
    # comment this line to use the best state ever found by that annealer
    # best_state, best_objective = annealer.get_best_state()
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
        time_check: bool = False,
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
        self.time_check = time_check

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
        self.send_queues = [ctx.Queue() for _ in range(self.n_chains)]
        self.receive_queues = [ctx.Queue() for _ in range(self.n_chains)]

        self.processes = []

        for send_queue, receive_queue, annealer_constructor in zip(
            self.send_queues, self.receive_queues, self.annealers_constructors
        ):
            process = ctx.Process(
                target=_worker, args=(send_queue, receive_queue, annealer_constructor)
            )
            process.start()
            self.processes.append(process)

        print(f"n_chains = {self.n_chains}")
        print(f"len(self.send_queues) = {len(self.send_queues)}")
        print(f"len(self.processes) = {len(self.processes)}")
        for i, (send_queue, receive_queue, proc) in enumerate(
            zip(self.send_queues, self.receive_queues, self.processes)
        ):
            print(
                f"Chain {i}: send_queue={send_queue}, receive_queue={receive_queue}, process pid={proc.pid}, alive={proc.is_alive()}"
            )

        # sys.exit()

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
            self.std = np.std(initial_energy)
            self.initial_temperature = max(
                self.initial_temperature, self.temp_params["min_initial_temp"]
            )
            self.temperature = self.initial_temperature

        else:
            self.temperature = self.initial_temperature

        # Initialize logger
        if self.temperature_schedule == "adaptive":
            fieldnames = [
                "step",
                "best_objective",
                "avg_acceptance_rate",
                "std",
                "temperature",
                "runtime",
            ]
        else:
            fieldnames = [
                "step",
                "best_objective",
                "avg_acceptance_rate",
                "temperature",
                "runtime",
            ]
        self.logger = CSVLogger(log_file, fieldnames=fieldnames)

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
        for send_queue in self.send_queues:
            send_queue.put(("get_current_objective", None))

        return [receive_queue.get() for receive_queue in self.receive_queues]

    def run_remote_chains(self, n_steps: int, seed: int = None):
        """
        Run the chains in the remote processes.
        """
        self.log_debug(f"running chains in parallel")
        for send_queue in self.send_queues:
            send_queue.put(("run_chain", {"n_steps": n_steps, "seed": seed}))

        self.log_debug(f"waiting for chains to finish")
        results = []
        for i, receive_queue in enumerate(self.receive_queues):
            try:
                result = receive_queue.get()
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

        for send_queue in self.send_queues:
            send_queue.put(("reset", None))

        return [receive_queue.get() for receive_queue in self.receive_queues]

    def set_state_remote_chains(self, states: List[Any], objectives: List[float]):
        """
        Set the state of the chains in the remote processes.
        """
        self.log_debug(f"=== SET_STATE_REMOTE_CHAINS START ===")
        self.log_debug(
            f"Number of states: {len(states)}, send_queues: {len(self.send_queues)}, receive_queues: {len(self.receive_queues)}, processes: {len(self.processes)}"
        )

        run_time = time.time() - self.start_time
        self.log_debug(f"run_time: {run_time} -- {self.max_runtime}")
        alives = {}
        for i, process in enumerate(self.processes):
            alive = process.is_alive()
            alives[i] = alive

        self.log_debug(f"alives: {alives}")

        outputs = []
        for i, (send_queue, receive_queue, process, state, objective) in enumerate(
            zip(
                self.send_queues,
                self.receive_queues,
                self.processes,
                states,
                objectives,
            )
        ):

            test_data = {"state": state, "objective": objective}
            pickled = pickle.dumps(test_data)
            self.log_debug(f"[{i}] Pickle successful, size: {len(pickled)} bytes")

            t = time.time()
            self.log_debug(
                f"{i}, about to send to remote process: {process.is_alive()} pid: {process.pid}"
            )
            # if receive_queue.poll(0):  # Non-blocking check
            #     self.log_debug(f"[{i}] WARNING: Unexpected data in pipe before send!")
            #     unexpected = remote.recv()
            #     self.log_debug(f"[{i}] Unexpected data: {unexpected}")

            send_queue.put(("set_state", {"state": state, "objective": objective}))
            self.log_debug(
                f"[{i}] Send completed, about to recv() process: {process.is_alive()}, pid: {process.pid}"
            )
            # try:

            #     # FIONREAD gets number of bytes in read buffer
            #     FIONREAD = termios.FIONREAD
            #     buf = array.array("i", [0])
            #     fcntl.ioctl(remote.fileno(), FIONREAD, buf)
            #     self.log_debug(f"Bytes in pipe buffer: {buf[0]}")
            # except:
            #     pass
            # self.log_debug(
            #     f"waiting for state to be set for remote {remote} {process.is_alive()}"
            # )
            outputs.append(receive_queue.get())

            self.log_debug(
                f"state set for remote {process.is_alive()} {time.time() - t} seconds"
            )

        self.log_debug(f"states set for all remote chains")

        # for i, remote in enumerate(self.remotes):
        #     self.log_debug(f"waiting for state to be set for remote {remote} {i}")
        #     outputs.append(remote.recv())
        #     self.log_debug(f"state set for remote {remote} {i}: {outputs[i]}")
        self.log_debug(f"=== SET_STATE_REMOTE_CHAINS END ===")
        return outputs

    def set_temperature_remote_chains(self, temperature: float):
        """
        Set the temperature of the chains in the remote processes.
        """
        for send_queue in self.send_queues:
            send_queue.put(("set_temperature", {"temperature": temperature}))

        return [receive_queue.get() for receive_queue in self.receive_queues]

    def get_temperature_remote_chains(self):
        """
        Get the temperature of the chains in the remote processes.
        """
        for send_queue in self.send_queues:
            send_queue.put(("get_temperature", None))

        return [receive_queue.get() for receive_queue in self.receive_queues]

    def clear_accepted_energy_remote_chains(self):

        for send_queue in self.send_queues:
            send_queue.put(("clear_accepted_energy", None))

        return [receive_queue.get() for receive_queue in self.receive_queues]

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

        try:
            for step in range(n_outer_iterations):
                begin_step_time = time.time()
                # Run chains in parallel
                self.log_debug(f"running chains in parallel")
                results = self.run_remote_chains(self.mixing_frequency)
                end_remote_chains_time = time.time()
                remote_chains_time = end_remote_chains_time - begin_step_time
                if self.time_check:
                    print(f"remote chains time: {remote_chains_time}")

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
                self._log_metrics(
                    step, best_states, best_objectives, acceptance_rates, self.std
                )
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
                begin_set_states_time = time.time()
                self.set_state_remote_chains(new_states, new_objectives)
                end_set_states_time = time.time()
                set_states_time = end_set_states_time - begin_set_states_time
                if self.time_check:
                    print(f"set states time: {set_states_time}")
                n_set_states += 1
                self.log_debug(f"states updated {n_set_states}")
                # update the temperature if adaptive
                if self.temperature_schedule == "adaptive":
                    s = 1 / self.temperature
                    rho = np.mean(acceptance_rates)
                    accepted_energies = np.concatenate(self.accepted_energies)
                    self.log_debug(f"accepted energies concatenated")

                    # print(f"original temperature: {self.temperature}")
                    # print(f"accepted energies: {accepted_energies}")
                    # print(f"rho: {rho}")

                    # make sure there are some accepted energies
                    if len(accepted_energies) > 0:
                        sig = np.std(accepted_energies)
                        # print(f"sig: {sig}")
                        self.std = sig
                        self.log_debug(f"standard deviation calculated")
                        # make sure the standard deviation isn't zero
                        if sig > 1e-8:
                            G = (4 * rho * (1 - rho) ** 2) / (2 - rho) ** 2
                            # print(f"G: {G}")
                            s_new = (
                                s
                                + self.temp_params["lambda"]
                                * (1 / sig)
                                * (1 / (s**2 * sig**2))
                                * G
                            )
                            # print(f"s_new: {s_new}")
                            self.temperature = max(1 / s_new, self.min_temperature)
                            self.log_debug(f"temperature updated")
                            # print(f"new temperature: {self.temperature}")
                            # input("Press Enter to continue...")

                    # clear the circular buffers (more efficient)
                    self.clear_accepted_energy_remote_chains()
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
            for i, send_queue in enumerate(self.send_queues):
                try:
                    send_queue.put(("close", None))
                except (BrokenPipeError, ConnectionResetError, OSError) as e:
                    print(f"Worker {i} connection already closed: {e}")
                    # Worker already closed
                    pass

            # Close all remote connections
            for send_queue in self.send_queues:
                try:
                    send_queue.close()
                except:
                    pass

            for receive_queue in self.receive_queues:
                try:
                    receive_queue.close()
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
        std: float = 0,
    ) -> None:
        """Log global metrics to CSV file."""
        if self.temperature_schedule == "adaptive":
            metrics = {
                "step": step,
                "best_objective": np.round(self.global_best_objective, 4),
                "avg_acceptance_rate": np.round(np.mean(acceptance_rates), 4),
                "std": np.round(std, 4),
                "temperature": np.round(self.temperature, 4),
                "runtime": np.round(time.time() - self.start_time, 4),
            }
        else:
            metrics = {
                "step": step,
                "best_objective": np.round(self.global_best_objective, 4),
                "avg_acceptance_rate": np.round(np.mean(acceptance_rates), 4),
                "temperature": np.round(self.temperature, 4),
                "runtime": np.round(time.time() - self.start_time, 4),
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
