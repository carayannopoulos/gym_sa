#!/usr/bin/env python3
"""
Example script demonstrating how to use the gym_sa package
to solve a Traveling Salesman Problem using simulated annealing.
"""

import numpy as np
from gym_sa import TSPEnv, Annealer, ParallelAnnealer

def single_chain_example():
    """Example using single-chain simulated annealing."""
    print("=== Single-Chain Simulated Annealing Example ===")

    # Create a TSP environment with 10 cities
    env_params = {
        "n_cities": 10,
        "seed": 42
    }

    # Create and run single-chain annealer
    annealer = Annealer(
        env_constructor=TSPEnv,
        env_params=env_params,
        initial_temperature=10.0,
        cooling_rate=0.995,
        min_temperature=1e-3,
        seed=42,
        verbose=False
    )

    # Run annealing for 1000 steps
    print("Running single-chain annealing...")
    for step in range(1000):
        annealer.step()

        if step % 200 == 0:
            best_state, best_objective = annealer.get_best_state()
            print(f"Step {step}: Best objective = {best_objective:.2f}")

    # Get final results
    best_state, best_objective = annealer.get_best_state()
    stats = annealer.get_stats()

    print(f"\nFinal Results:")
    print(f"Best tour length: {best_objective:.2f}")
    print(f"Best tour: {best_state}")
    print(f"Acceptance rate: {stats['mean_acceptance_rate']:.3f}")
    print(f"Final temperature: {annealer.temperature:.6f}")

def parallel_chain_example():
    """Example using parallel simulated annealing."""
    print("\n=== Parallel Simulated Annealing Example ===")

    # Create a TSP environment with 15 cities
    env_params = {
        "n_cities": 15,
        "seed": 42
    }

    # Create and run parallel annealer
    parallel_annealer = ParallelAnnealer(
        env_constructor=TSPEnv,
        env_params=env_params,
        n_chains=4,
        initial_temperature=10.0,
        cooling_rate=0.995,
        min_temperature=1e-3,
        mixing_frequency=50,
        max_steps=500,
        base_seed=42,
        verbose=True,
        log_file="parallel_annealing_example.csv"
    )

    print("Running parallel annealing...")
    results = parallel_annealer.run()

    print(f"\nFinal Results:")
    print(f"Best tour length: {results['best_objective']:.2f}")
    print(f"Best tour: {results['best_state']}")
    print(f"Total steps: {results['total_steps']}")
    print(f"Runtime: {results['runtime']:.2f} seconds")

def compare_methods():
    """Compare single-chain vs parallel annealing."""
    print("\n=== Method Comparison ===")

    env_params = {"n_cities": 12, "seed": 42}

    # Single-chain results
    annealer = Annealer(
        env_constructor=TSPEnv,
        env_params=env_params,
        initial_temperature=10.0,
        cooling_rate=0.995,
        seed=42
    )

    for _ in range(500):
        annealer.step()

    single_best_state, single_best_objective = annealer.get_best_state()

    # Parallel results
    parallel_annealer = ParallelAnnealer(
        env_constructor=TSPEnv,
        env_params=env_params,
        n_chains=4,
        max_steps=500,
        base_seed=42
    )

    parallel_results = parallel_annealer.run()

    print(f"Single-chain best objective: {single_best_objective:.2f}")
    print(f"Parallel best objective: {parallel_results['best_objective']:.2f}")
    print(f"Improvement: {((single_best_objective - parallel_results['best_objective']) / single_best_objective * 100):.1f}%")

if __name__ == "__main__":
    print("Gym-SA Package Example")
    print("=" * 50)

    try:
        # Run examples
        single_chain_example()
        parallel_chain_example()
        compare_methods()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
