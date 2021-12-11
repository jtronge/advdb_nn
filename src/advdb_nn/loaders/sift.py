"""Loaders for the SIFT dataset.

See http://corpus-texmex.irisa.fr/.
"""
import numpy as np


def read_fvecs(fname):
    """Read a SIFT fvecs data file."""
    data = np.fromfile(fname, dtype='int32')
    dim = data[0]
    data = np.reshape(data, (-1, dim + 1))[:, 1:]
    return data.view('float32')


def read_ivecs(fname):
    """Read a SIFT ivecs data file."""
    data = np.fromfile(fname, dtype='int32')
    dim = data[0]
    return np.reshape(data, (-1, dim + 1))[:, 1:]
