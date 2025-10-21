"""
Gym-SA: Simulated Annealing for Optimization Problems

A Python package that implements simulated annealing algorithms for optimization
problems using Gymnasium environments.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes
from .annealer import Annealer
from .gym_psa import ParallelAnnealer
from .tsp_env import TSPEnv
from .utils import softmax, set_seed
from .logger import CSVLogger

# Import benchmark modules
# from . import tsp_benchmark
# from . import psa_benchmark

__all__ = [
    "Annealer",
    "ParallelAnnealer",
    "TSPEnv",
    "softmax",
    "set_seed",
    "CSVLogger",
    # "tsp_benchmark",
    # "psa_benchmark",
]
