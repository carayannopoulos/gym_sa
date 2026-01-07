import numpy as np
from copy import deepcopy
from collections import deque
import os
import ray
from .utils import softmax, set_seed
from typing import Callable, Optional, Tuple, Any, List, Dict


@ray.remote(num_cpus=0.05)
class SA_RolloutWorker:
    def __init__(
        self,
        env_constructor: callable = None,
        env_params: dict = None,
        temperature_schedule: str = "geometric",
        initial_temperature=1.0,
        cooling_rate=0.99,
        min_temperature=1e-3,
        seed=None,
        verbose=False,
        obj_dict=None,
    ):
        """
        Initialize the Annealer.
        Args:
            env: An instance of the custom Gymnasium environment.
            initial_temperature (float): Starting temperature for annealing.
            cooling_rate (float): Multiplicative factor for temperature decay.
            min_temperature (float): Minimum temperature to stop cooling.
            seed (int, optional): Random seed for reproducibility.
        """

        if obj_dict is not None:
            self.initialize_from_dict(obj_dict)
        else:
            self.env = env_constructor(**env_params)
            self.temperature = initial_temperature
            self.cooling_rate = cooling_rate
            self.min_temperature = min_temperature
            self.rng = np.random.default_rng(seed)
            self.verbose = verbose
            self.temperature_schedule = temperature_schedule
            if temperature_schedule == "adaptive":
                # self.accepted_energy = []  # Old: growing list
                self.accepted_energy = deque(
                    maxlen=1000
                )  # New: fixed-size circular buffer

            # Stats
            self.accepted = 0
            self.total = 0
            # self.acceptance_history = []  # Old: growing list
            self.acceptance_history = deque(
                maxlen=1000
            )  # New: fixed-size circular buffer

            self.debug_file = os.path.join(self.env.agent_dir, "annealer_debug.txt")
            with open(self.debug_file, "w") as f:
                f.write(f"Annealer debug:\n")

            self.n_set_state = 0
            self.else_set_state = 0

    def log_debug(self, message):
        with open(self.debug_file, "a") as f:
            f.write(f"{message}\n")

    def initialize_from_dict(self, obj_dict):
        self.env = obj_dict["env"]
        self.temperature = obj_dict["temperature"]
        self.temperature_schedule = obj_dict["temperature_schedule"]
        self.cooling_rate = obj_dict["cooling_rate"]
        self.min_temperature = obj_dict["min_temperature"]
        self.rng = obj_dict["rng"]
        self.verbose = obj_dict["verbose"]
        self.accepted = obj_dict["accepted"]
        self.total = obj_dict["total"]
        # Handle both old list format and new deque format for backward compatibility
        acceptance_history_data = obj_dict["acceptance_history"]
        if isinstance(acceptance_history_data, list):
            # Convert old list to new deque format
            self.acceptance_history = deque(acceptance_history_data, maxlen=1000)
        else:
            self.acceptance_history = acceptance_history_data
        self.state = obj_dict["state"]
        self.current_objective = obj_dict["current_objective"]
        self.best_state = obj_dict["best_state"]
        self.best_objective = obj_dict["best_objective"]
        if self.temperature_schedule == "adaptive":
            # Handle both old list format and new deque format for backward compatibility
            accepted_energy_data = obj_dict["accepted_energy"]
            if isinstance(accepted_energy_data, list):
                # Convert old list to new deque format
                self.accepted_energy = deque(accepted_energy_data, maxlen=1000)
            else:
                self.accepted_energy = accepted_energy_data

    def step(self):
        """
        Perform one simulated annealing step: propose, evaluate, accept/reject, update stats.
        """
        self.log_debug(f"beginning of step... getting perturbation")
        # get the perturbation
        perturbation = self.env.get_perturbation(self.temperature)
        if self.verbose:
            print(f"Perturbation: {perturbation}")
        self.log_debug(f"perturbation: {perturbation}")
        # apply the perturbation
        next_state, next_objective, done, info = self.env.step(perturbation)
        if self.verbose:
            print(f"self.state: {self.state} -- next_state: {next_state}")
        self.log_debug(f"perturbation applied")
        # calculate the delta
        delta = next_objective - self.current_objective
        if self.verbose:
            print(f"delta: {delta}")

        # accept/reject
        accept = False
        if delta >= 0:
            accept = True
        else:
            prob = np.exp(delta / max(self.temperature, 1e-8))
            if self.verbose:
                print(f"prob: {prob} -- temperature: {self.temperature}")
            if self.rng.uniform() < prob:
                accept = True
        self.total += 1
        if self.verbose:
            print(f"accept: {accept}")

        if accept:
            # update the state
            self.state = deepcopy(next_state)
            # update the objective
            self.current_objective = next_objective
            # update the acceptance count
            self.accepted += 1
            # update the accepted energy
            if self.temperature_schedule == "adaptive":
                # self.accepted_energy.append(self.current_objective)  # Old: growing list
                self.accepted_energy.append(
                    self.current_objective
                )  # New: fixed-size circular buffer

            # check if the new state is better than the best state
            if next_objective > self.best_objective:
                # update the best state
                self.best_state = deepcopy(next_state)
                # update the best objective
                self.best_objective = next_objective

        else:
            self.else_set_state += 1
            self.log_debug(f"before else set state {self.else_set_state}")
            self.env.set_state(self.state, self.current_objective)
            self.log_debug(f"after else set state {self.else_set_state}")

        # Cool down
        if self.temperature_schedule == "geometric":
            self.temperature = max(
                self.temperature * self.cooling_rate, self.min_temperature
            )
        elif self.temperature_schedule == "adaptive":
            pass

        if self.verbose:
            print(f"new state: {self.state} -- new objective: {self.current_objective}")

        # Track acceptance rate
        # self.acceptance_history.append(1 if accept else 0)  # Old: growing list
        self.acceptance_history.append(
            1 if accept else 0
        )  # New: fixed-size circular buffer
        return accept

    def reset(self):
        self.state, self.current_objective = self.env.reset()
        self.best_state = deepcopy(self.state)
        self.best_objective = self.current_objective

    def set_state(self, state, objective):
        # self.log_debug(f"setting state in annealer {self.n_set_state}")
        self.state = deepcopy(state)
        # self.log_debug(f"state copied {self.n_set_state}")
        self.current_objective = objective
        # self.log_debug(f"objective set {self.n_set_state}")
        self.env.set_state(state, objective)
        # self.log_debug(f"state set in env {self.n_set_state}")
        self.n_set_state += 1

    def set_temperature(self, temperature):
        self.temperature = temperature

    def get_temperature(self):
        return self.temperature

    def clear_accepted_energy(self):
        self.accepted_energy.clear()

    def get_current_objective(self):
        return self.env.current_objective

    def get_best_state(self):
        """
        Return the best state and objective found so far.
        """
        return deepcopy(self.best_state), self.best_objective

    def get_stats(self, period=100):
        """
        Return statistics for logging.
        Args:
            period (int): Number of steps to average acceptance rate over.
        Returns:
            dict: Stats including best objective, current temperature, mean acceptance rate.
        """
        # Old: slice operation on growing list
        # recent_accepts = (
        #     self.acceptance_history[-period:] if period > 0 else self.acceptance_history
        # )
        # mean_acceptance = np.mean(recent_accepts) if recent_accepts else 0.0

        # New: use all available data in circular buffer (up to maxlen)
        if period > 0 and len(self.acceptance_history) > period:
            # Take only the last 'period' elements from the circular buffer
            recent_accepts = list(self.acceptance_history)[-period:]
        else:
            # Use all available data in the circular buffer
            recent_accepts = list(self.acceptance_history)
        mean_acceptance = np.mean(recent_accepts) if recent_accepts else 0.0
        return {
            "best_objective": np.round(self.best_objective, 2),
            "current_temperature": np.round(self.temperature, 2),
            "mean_acceptance_rate": np.round(mean_acceptance, 2),
            "accepted": self.accepted,
            "total": self.total,
        }

    def run_chain(
        self, n_steps: int, seed: int = None
    ) -> Tuple[Any, float, float, Any]:
        """
        Run a single annealing chain.

        Args:
            n_steps: Number of steps to run the chain
            seed: Random seed for this chain

        Returns:
            Tuple of (best_state, best_objective, acceptance_rate, current_state)
        """
        # Set seed for this process
        if seed is not None:
            set_seed(seed)

        # Run the chain for n_steps
        for _ in range(n_steps):
            self.step()

        # Get final stats
        best_state, best_objective = self.get_best_state()
        stats = self.get_stats()
        acceptance_rate = stats["mean_acceptance_rate"]

        return (
            self.accepted_energy,
            best_state,
            best_objective,
            acceptance_rate,
            self.state,
            self.current_objective,
        )

    def ping(self):
        print("Ping from worker", flush=True)
        return "pong"
