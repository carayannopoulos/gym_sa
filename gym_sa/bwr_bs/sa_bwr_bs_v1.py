from bwr_bs.envs.bwr_bs_sa_v1 import bwr_bs_sa_v1 as bwr
import numpy as np
from gym_sa.gym_psa_v5 import ParallelAnnealer
import pandas as pd
import matplotlib.pyplot as plt
import time
import faulthandler, os, sys


def main(
    n_chains: int = 2,
    initial_temperature: float = 10,
    cooling_rate: float = 0.995,
    min_temperature: float = 0.01,
    mixing_frequency: int = 50,
    max_steps: int = 2000,
    min_acceptance_rate: float = -1,
    max_runtime: float = 3600,
    base_seed: int = 42,
    log_file: str = "test_log.csv",
    env_params: dict = None,
    temperature_schedule: str = "geometric",
    temp_params: dict = None,
    verbose: bool = False,
    save_file: str = "save_dict.pkl",
    save_progress: bool = False,
    save_freq: int = None,
    load_file: str = None,
    load_progress: bool = False,
):

    # Initialize parallel annealer with 4 chains
    parallel_annealer = ParallelAnnealer(
        env_constructor=bwr,
        env_params=env_params,
        n_chains=n_chains,
        temperature_schedule=temperature_schedule,
        temp_params=temp_params,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        mixing_frequency=mixing_frequency,  # Mix states every 50 steps
        max_steps=max_steps,
        min_acceptance_rate=min_acceptance_rate,
        max_runtime=max_runtime,
        base_seed=base_seed,
        log_file=log_file,
        verbose=verbose,
        save_file=save_file,
        save_progress=save_progress,
        save_freq=save_freq,
        load_file=load_file,
        load_progress=load_progress,
    )

    # Run parallel annealing
    best_state, best_objective = parallel_annealer.run()

    # print(f"\nBest state found: {best_state}")
    # print(f"Best objective found: {best_objective}")

    # # Verify that we found a good solution
    # # For MockEnv, the optimal state is 3.0 with objective 0.0
    # print(f"\nDistance from optimal state (3.0): {abs(best_state - 3.0)}")
    # print(f"Distance from optimal objective (0.0): {abs(best_objective - 0.0)}")

    return best_objective


if __name__ == "__main__":
    # Enable faulthandler to get stack traces on hard crashes / deadlocks
    try:
        log_path = os.path.join(os.path.dirname(__file__), "faulthandler_main.log")
        with open(log_path, "w") as fh:
            faulthandler.enable(fh)
            # Also dump tracebacks if the process hangs for > 300s
            faulthandler.dump_traceback_later(300, repeat=True, file=fh)
    except Exception:
        pass

    env_params = {
        "env_data": "bs_input.xlsx",
        "exepath": "/home/loukas/Simulate3/bin/simulate3",
    }

    total_steps = 50000000
    n_chains = 32
    n_steps = total_steps  # // n_chains
    initial_temperature = 200
    cooling_rate = 0.9999
    min_temperature = 0.00001
    mixing_frequency = 20
    min_acceptance_rate = -1
    max_runtime = 3600000
    base_seed = 42
    temperature_schedule = "adaptive"
    temp_params = {"lambda": 1, "alpha": 2, "min_initial_temp": 0.000001}

    main(
        n_chains=n_chains,
        env_params=env_params,
        max_steps=n_steps,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        min_temperature=min_temperature,
        mixing_frequency=mixing_frequency,
        temperature_schedule=temperature_schedule,
        temp_params=temp_params,
        max_runtime=max_runtime,
        verbose=True,
        log_file="bwr_bs_v1_log.csv",
        save_file="save_dict.pkl",
        save_progress=True,
        save_freq=3,
        # load_file="save_dict.pkl",
        load_progress=False,
    )
