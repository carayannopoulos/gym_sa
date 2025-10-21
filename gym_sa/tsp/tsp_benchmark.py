import numpy as np
from gym_sa.gym_psa_v3 import ParallelAnnealer
from gym_sa.tsp_env import TSPEnv
import pandas as pd
import matplotlib.pyplot as plt
import time


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
):

    # Initialize parallel annealer with 4 chains
    parallel_annealer = ParallelAnnealer(
        env_constructor=TSPEnv,
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


def obj_func(x):
    dim = 10
    # return -(10 * dim + sum(x**2 - 10 * np.cos(2 * np.pi * x)))
    return -sum(x**2)


if __name__ == "__main__":

    # ch150 benchmark
    cities = np.loadtxt("brd14051.csv", delimiter="\t")
    # cities = np.loadtxt("ch150_nodes.csv", delimiter=",")
    dim = cities.shape[1]
    print(cities)

    env_params = {
        "dim": dim,
        "lims": (0, 1),
        "n_cities": cities.shape[0],
        "init_cities": "coordinates",
        "cities": cities,
        "render_best_state": False,
        "render_delay": 5000000000000,
    }
    env_params = {
        "env_params": env_params,
    }

    total_steps = 2000000
    n_chains = 32
    n_steps = total_steps  # // n_chains
    initial_temperature = 200
    cooling_rate = 0.9999
    min_temperature = 0.00001
    mixing_frequency = 20
    min_acceptance_rate = -1
    max_runtime = 3600
    base_seed = 42
    # temperature_schedule = "geometric"
    # temp_params = {}
    verbose = True
    temperature_schedule = "adaptive"
    temp_params = {"lambda": 0.5, "alpha": 2, "min_initial_temp": 5}

    n_its = 1
    best_objectives = []
    for i in range(n_its):
        print(f"Iteration {i+1} of {n_its}")
        best_objective = main(
            n_chains=n_chains,
            env_params=env_params,
            max_steps=n_steps,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            mixing_frequency=mixing_frequency,
            temperature_schedule=temperature_schedule,
            temp_params=temp_params,
            verbose=verbose,
        )
        best_objectives.append(best_objective)

    best_objectives_mean = np.mean(best_objectives)
    best_objectives_std = np.std(best_objectives)

    print(f"Best objective: {best_objectives_mean:.4f} ± {best_objectives_std:.4f}")

    df = pd.read_csv("test_log.csv")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5))

    # Plot best objective
    ax1.plot(df["step"], -df["best_objective"], label=f"n_chains={n_chains}")
    ax1.set_ylabel("Best Objective")
    ax1.set_xlabel("Step")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True)

    # Plot acceptance rate and temperature
    ax2_temp = ax2.twinx()

    # Plot acceptance rate on left y-axis
    ax2.plot(df["step"], df["avg_acceptance_rate"], "g-", label=f"n_chains={n_chains}")
    ax2.set_ylabel("Acceptance Rate", color="g")
    ax2.tick_params(axis="y", labelcolor="g")

    # Plot temperature on right y-axis
    ax2_temp.plot(df["step"], df["temperature"], "r-", label=f"n_chains={n_chains}")
    ax2_temp.set_ylabel("Temperature", color="r")
    ax2_temp.tick_params(axis="y", labelcolor="r")

    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_temp.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax2.set_xlabel("Step")
    ax2.grid(True)

    plt.tight_layout()
    # plt.savefig("test_parallel_annealer.png")
    plt.show()
