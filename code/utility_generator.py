import scipy.stats as sp
import numpy as np


def generate_normal_utilities_agent_based(loc: float = 0, scale: float = 1, n: int = 10) -> np.ndarray:
    """
    Generate a random utility for a single agent using a normal distribution to randomly draw the utility of each item.

    :param loc: the float representing the mean of the normal distribution.
    :param scale: the float representing the standard deviation of the normal distribution.
    :param n: the number of items.
    :return: an array which represents the utility function for the agent.
    """
    rv = sp.norm(loc, scale)

    return rv.rvs(n)


def generate_uniform_utilities_agent_based(loc: float = -1, scale: float = 2, n: int = 10) -> np.ndarray:
    """
    Generate a random utility for a single agent using a uniform distribution to randomly draw the utility of each item.

    :param loc: the float representing minimum utility for an item.
    :param scale: the float such that loc + scale is the maximum utility for an item.
    :param n: the number of items.
    :return: an array which represents the utility function for the agent.
    """
    rv = sp.uniform(loc, scale)

    return rv.rvs(n)
