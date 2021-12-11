"""Data set."""
import pickle


class Dataset:
    """Data set representation."""
    # TODO: This takes a lot of memory.

    def __init__(self, X, Q, top_k, truth):
        """Data set constructor."""
        self.X = X
        self.Q = Q
        self.top_k = top_k
        self.truth = truth

    def dump(self, fname):
        """Dump a representation of this object to a file."""
        with open(fname, 'wb') as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(fname):
        """Load a representation of this object from the file."""
        with open(fname, 'rb') as fp:
            return pickle.load(fp)
