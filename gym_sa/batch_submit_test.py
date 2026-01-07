from bwr_bs.envs.bwr_bs_sa_v2 import bwr_bs_sa_v2 as bwr
import numpy as np
from gym_sa import DistributedAnnealer
import pandas as pd
import matplotlib.pyplot as plt
import time
import faulthandler, os, sys
import argparse, json
import shutil


def main(
    runtime_env: dict = None,
    ray_address: str = None,
    working_dir: str = ".",
    n_chains: int = 60,
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
    distributed_annealer = DistributedAnnealer(
        runtime_env=runtime_env,
        ray_address=ray_address,
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
    best_state, best_objective = distributed_annealer.run()

    # print(f"\nBest state found: {best_state}")
    # print(f"Best objective found: {best_objective}")

    # # Verify that we found a good solution
    # # For MockEnv, the optimal state is 3.0 with objective 0.0
    # print(f"\nDistance from optimal state (3.0): {abs(best_state - 3.0)}")
    # print(f"Distance from optimal objective (0.0): {abs(best_objective - 0.0)}")

    return best_objective


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    import datetime
    import os

    def parse_json(value: str):
        try:
            return json.loads(value)
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}")

    # Defaults based on current file values
    default_runtime_env = {
        # "working_dir": result_dir,
        "conda": "ML",
        "env_vars": {
            "PYTHONPATH": current_dir,
            # Ensure each Ray worker stays single-threaded for BLAS/OpenMP
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "excludes": ["*.res"],
    }

    default_env_params = {
        "env_data": "bs_input.xlsx",
        "exepath": "/home/loukas/Simulate3/bin/simulate3",
        # "working_dir": result_dir,
        "archive_dir": "bwr_archive/",
    }

    default_temp_params = {"lambda": 1, "alpha": 1, "min_initial_temp": 0.000001}

    parser = argparse.ArgumentParser(
        description="Distributed Simulated Annealing runner"
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--n-chains", type=int, default=60)
    parser.add_argument("--total-steps", type=int, default=50000000)
    parser.add_argument("--initial-temperature", type=float, default=200)
    parser.add_argument("--cooling-rate", type=float, default=0.9999)
    parser.add_argument("--min-temperature", type=float, default=0.00001)
    parser.add_argument("--mixing-frequency", type=int, default=10)
    parser.add_argument("--min-acceptance-rate", type=float, default=-1)
    parser.add_argument("--max-runtime", type=float, default=3600000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--load-file", default=None)
    parser.add_argument("--save-freq", type=int, default=3)
    parser.add_argument(
        "--temperature-schedule",
        default="adaptive",
        choices=["adaptive", "geometric", "linear", "exponential"],
    )  # adjust if needed
    parser.add_argument(
        "--temp-params",
        type=parse_json,
        default=default_temp_params,
        help="JSON string for temperature parameters",
    )
    # Optional explicit temperature param overrides
    parser.add_argument(
        "--temp-lambda", type=float, default=None, help="Override temperature lambda"
    )
    parser.add_argument(
        "--temp-alpha", type=float, default=None, help="Override temperature alpha"
    )
    parser.add_argument(
        "--temp-min-initial-temp",
        type=float,
        default=None,
        help="Override temperature min_initial_temp",
    )
    parser.add_argument(
        "--env-params",
        type=parse_json,
        default=default_env_params,
        help="JSON string for environment parameters",
    )
    parser.add_argument(
        "--runtime-env",
        type=parse_json,
        default=default_runtime_env,
        help="JSON string for Ray runtime_env",
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save-progress", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--load-progress", action=argparse.BooleanOptionalAction, default=True
    )

    # Optional fine-grained overrides for env_params
    parser.add_argument("--env-data", default=None)
    parser.add_argument("--exepath", default=None)
    parser.add_argument("--archive-dir", default=None)

    args = parser.parse_args()

    # Resolve dictionaries and apply overrides
    runtime_env = args.runtime_env
    env_params = args.env_params
    temp_params = args.temp_params
    # Apply explicit temp_* overrides if provided
    if args.temp_lambda is not None:
        temp_params["lambda"] = args.temp_lambda
    if args.temp_alpha is not None:
        temp_params["alpha"] = args.temp_alpha
    if args.temp_min_initial_temp is not None:
        temp_params["min_initial_temp"] = args.temp_min_initial_temp

    # Working directory propagation
    if args.working_dir is not None:
        runtime_env["working_dir"] = args.working_dir
        env_params["working_dir"] = args.working_dir

    if args.env_data is not None:
        env_params["env_data"] = args.env_data
    if args.exepath is not None:
        env_params["exepath"] = args.exepath
    if args.archive_dir is not None:
        env_params["archive_dir"] = args.archive_dir
    if args.archive_dir is not None:
        env_params["archive_dir"] = args.archive_dir

    n_steps = args.total_steps  # keep behavior consistent

    current_dir = os.path.dirname(os.path.abspath(__file__))
    archive_dir = "/orcd/home/002/loukas/phd/so_algs/dsa/hyperparameter_tuning/cases"
    now = datetime.datetime.now().date()
    result_dir = os.path.join(
        archive_dir,
        f"restart_lambda_{args.temp_lambda}_alpha_{args.temp_alpha}_mixing_{args.mixing_frequency}_{now}",
    )

    print(f"Result directory: {result_dir}")
    os.makedirs(result_dir, exist_ok=True)

    scratch_dir = "/orcd/home/002/loukas/orcd/scratch"
    working_dir = os.path.join(
        scratch_dir,
        f"lambda_{args.temp_lambda}_alpha_{args.temp_alpha}_mixing_{args.mixing_frequency}_{now}",
    )

    os.makedirs(working_dir, exist_ok=True)
    shutil.copy(env_params["env_data"], working_dir)

    env_params["archive_dir"] = result_dir
    runtime_env["working_dir"] = working_dir

    log_file = os.path.join(result_dir, "log.csv")
    save_file = os.path.join(result_dir, "save_dict.pkl")

    main(
        runtime_env=runtime_env,
        ray_address=args.ray_address,
        n_chains=args.n_chains,
        env_params=env_params,
        max_steps=n_steps,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        min_temperature=args.min_temperature,
        mixing_frequency=args.mixing_frequency,
        temperature_schedule=args.temperature_schedule,
        temp_params=temp_params,
        max_runtime=args.max_runtime,
        verbose=args.verbose,
        log_file=log_file,
        save_file=save_file,
        save_progress=args.save_progress,
        save_freq=args.save_freq,
        load_file=args.load_file,
        load_progress=args.load_progress,
    )
