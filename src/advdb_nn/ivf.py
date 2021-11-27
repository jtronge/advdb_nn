import os
import sys
import numpy as np
from advdb_nn.util import dist2


class IVF:
    """IVF."""

    def __init__(self, X, c_count):
        """IVF constructor."""
        self.X = X
        # Initially choose random centroids
        centroids = np.random.randint(0, X.shape[0], c_count)

        # Build the inverted index
        self.ivf = []
        for i in range(c_count):
            self.ivf.append([])
        for x_i, x in enumerate(X):
            min_d = None
            min_c_i = None
            for c_i, c_x_i in enumerate(centroids):
                c = X[c_x_i]
                d = dist2(x, c)
                if min_d is None or min_c_i is None or d < min_d:
                    min_d = d
                    min_c_i = c_i
            assert min_c_i is not None
            # Add this dataset vector to the inverted index
            self.ivf[min_c_i].append(x_i)

        # Recalculate the centroids based on the closest dataset vectors
        self.centroids = []
        for c_i, c_ivf in enumerate(self.ivf):
            if c_ivf:
                c = sum(self.X[x_i] for x_i in c_ivf) / len(c_ivf)
            else:
                c = self.X[centroids[c_i]]
            self.centroids.append(c)

    def query(self, q, top_k):
        """Query the top-k vectors closest to q."""
        # Find the closest centroid
        min_d = None
        min_c_i = None
        for c_i, c in enumerate(self.centroids):
            d = dist2(q, c)
            if min_d is None or min_c_i is None or d < min_d:
                min_d = d
                min_c_i = c_i
        assert min_c_i is not None
        # Now compute the distances between q and all those in the inverted
        # index
        x_dist = [(x, dist2(self.X[x], q)) for x in self.ivf[min_c_i]]
        x_dist.sort(key=lambda x_d: x_d[1])
        assert len(x_dist) >= top_k
        return [x_d[0] for x_d in x_dist[:top_k]]
