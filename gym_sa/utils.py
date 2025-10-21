import numpy as np
import random


def softmax(logits, temperature=1.0):
    """
    Compute softmax values for each set of scores in logits.
    Args:
        logits (list or np.ndarray): Input values (e.g., rewards/objectives).
        temperature (float): Softmax temperature parameter.
    Returns:
        np.ndarray: Softmax probabilities.
    """
    logits = np.array(logits, dtype=np.float64)
    logits = logits / temperature
    e_logits = np.exp(logits - np.max(logits))  # for numerical stability
    return e_logits / np.sum(e_logits)


def set_seed(seed):
    """
    Set random seed for reproducibility (numpy and random).
    Args:
        seed (int): The seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
