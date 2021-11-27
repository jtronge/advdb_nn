import os
import sys
import numpy as np


def dist2(a, b):
    """Squared distance calculation."""
    u = b - a
    return np.dot(u, u)


def recall(expected, result):
    """Calculate the recall for a correct and result."""
    exp_s = set(expected)
    res_s = set(result)
    return len(exp_s.intersection(res_s)) / len(exp_s)
