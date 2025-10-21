# Gym-SA: Simulated Annealing for Optimization Problems

A Python package that implements simulated annealing algorithms for optimization problems using Gymnasium environments. This package provides both single-chain and parallel simulated annealing implementations, with support for various optimization problems including the Traveling Salesman Problem (TSP).

## Features

- **Single-chain Simulated Annealing**: Classic simulated annealing implementation with configurable temperature schedules
- **Parallel Simulated Annealing**: Multi-chain parallel implementation for improved exploration
- **Gymnasium Integration**: Built on top of Gymnasium for standardized environment interfaces
- **TSP Environment**: Built-in Traveling Salesman Problem environment
- **Flexible Temperature Schedules**: Support for geometric, adaptive, and custom temperature schedules
- **Comprehensive Logging**: Built-in logging and benchmarking tools
- **Extensible Design**: Easy to extend for new optimization problems

## Installation

### From Source

```bash
git clone https://github.com/yourusername/gym-sa.git
cd gym-sa
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/gym-sa.git
cd gym-sa
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from gym_sa import Annealer, ParallelAnnealer
from gym_sa import TSPEnv

# Create a TSP environment
env_params = {
    "n_cities": 20,
    "seed": 42
}

# Single-chain annealing
annealer = Annealer(
    env_constructor=TSPEnv,
    env_params=env_params,
    initial_temperature=1.0,
    cooling_rate=0.99,
    min_temperature=1e-3
)

# Run annealing
for step in range(1000):
    annealer.step()

best_state, best_objective = annealer.get_best_state()
print(f"Best objective: {best_objective}")

# Parallel annealing
parallel_annealer = ParallelAnnealer(
    env_constructor=TSPEnv,
    env_params=env_params,
    n_chains=4,
    max_steps=1000
)

results = parallel_annealer.run()
print(f"Best objective: {results['best_objective']}")
```

### TSP Example

```python
from gym_sa import TSPEnv, ParallelAnnealer

# Create TSP environment
env = TSPEnv(n_cities=50, seed=42)

# Run parallel simulated annealing
annealer = ParallelAnnealer(
    env_constructor=TSPEnv,
    env_params={"n_cities": 50, "seed": 42},
    n_chains=8,
    max_steps=5000,
    initial_temperature=10.0,
    cooling_rate=0.995
)

results = annealer.run()
print(f"Best tour length: {results['best_objective']}")
print(f"Best tour: {results['best_state']}")
```

## API Reference

### Core Classes

#### Annealer

The main simulated annealing class for single-chain optimization.

```python
Annealer(
    env_constructor: callable,
    env_params: dict,
    temperature_schedule: str = "geometric",
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.99,
    min_temperature: float = 1e-3,
    seed: int = None,
    verbose: bool = False
)
```

#### ParallelAnnealer

Multi-chain parallel simulated annealing implementation.

```python
ParallelAnnealer(
    env_constructor: callable,
    env_params: dict,
    n_chains: int,
    temperature_schedule: str = "geometric",
    temp_params: dict = None,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.99,
    min_temperature: float = 1e-3,
    mixing_frequency: int = 100,
    max_steps: int = 1000,
    min_acceptance_rate: float = 0.01,
    max_runtime: float = 3600,
    base_seed: int = None,
    verbose: bool = False,
    log_file: str = "parallel_annealing_log.csv"
)
```

#### TSPEnv

Traveling Salesman Problem environment.

```python
TSPEnv(
    n_cities: int,
    seed: int = None,
    distance_matrix: np.ndarray = None
)
```

## Temperature Schedules

The package supports several temperature schedules:

- **Geometric**: `T(t) = T_0 * (cooling_rate)^t`
- **Adaptive**: Automatically adjusts temperature based on acceptance rate
- **Custom**: User-defined temperature functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{gym_sa,
  title={Gym-SA: Simulated Annealing for Optimization Problems},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gym-sa}
}
```
