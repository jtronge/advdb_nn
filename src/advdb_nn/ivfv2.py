import os
import sys
import numpy as np
from advdb_nn.util import dist2


class Codebook:
    """Codebook representation."""

    def __init__(self, data, c_count):
        """Create the codebook."""
        # Initially choose random centroids
        centroids = np.random.randint(0, data.shape[0], c_count)

        # Build the inverted index
        self.ivf = []
        for i in range(c_count):
            self.ivf.append([])
        for x_i, x in enumerate(data):
            min_d = None
            min_c_i = None
            for c_i, c_x_i in enumerate(centroids):
                c = data[c_x_i]
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
                c = sum(data[x_i] for x_i in c_ivf) / len(c_ivf)
            else:
                c = data[centroids[c_i]]
            self.centroids.append(c)

    def quantize(self, v):
        """Return the index of the closest centroid (quantize the vector)."""
        min_d2 = None
        min_c_i = None
        for c_i, c in enumerate(self.centroids):
            d2 = dist2(v, c)
            if min_d2 is None or min_c_i is None or d2 < min_d2:
                min_d2 = d2
                min_c_i = c_i
        return min_c_i


class IVFv2:
    """IVF."""

    def __init__(self, X, c_count, m=8, bits=3):
        """IVF constructor."""
        # self.X = X
        self.ivf_codebook = Codebook(X, c_count)

        # Quantize and train on the subvectors
        self.m = m
        assert X.shape[1] % m == 0
        self.d = int(X.shape[1] / m)
        self.codebooks = [Codebook(X[:,i * self.d:(i + 1) * self.d], 2 ** bits)
                          for i in range(m)]

        # Compute the quantized data
        self.quantized_data = [
            [self.codebooks[i].quantize(x[i * self.d:(i + 1) * self.d]) for i in range(self.m)]
            for x in X
        ]
        # TODO: Compute the quantized X

    def subvectors(self, data, i):
        """Split the data vector into the ith subvectors of each entry."""
        return data[:,i * self.d:(i + 1) * self.d]

    def subvector(self, v, i):
        """Compute the subvector ith subvector of v."""
        return v[i * self.d:(i + 1) * self.d]

    def query(self, q, top_k, c_search=1):
        """Query the top-k vectors closest to q.

        :param top_k: number of result indices to return in a list
        :param c_search: number of centroids to search
        """
        # Find the closest `c_search` centroids to q
        close = [(c_i, dist2(q, c))
                 for c_i, c in enumerate(self.ivf_codebook.centroids)]
        close.sort(key=lambda c: c[1])
        close = [c[0] for c in close[:c_search]]

        # Compute the subvectors of q
        q_s = [self.subvector(q, i) for i in range(self.m)]
        # Now precompute distances between q and all the inner codebooks
        precompute = [
            [dist2(q_s[i], c) for c in cb.centroids]
            for i, cb in enumerate(self.codebooks)
        ]

        # Now calculate the "distances"
        x_dist = [
            (x_i, sum(precompute[i][v]
                      for i, v in enumerate(self.quantized_data[x_i])))
            for close_i in close for x_i in self.ivf_codebook.ivf[close_i]
        ]
        # Now compute the distances between q and all those closest to the given
        # centroids in the index
        # x_dist = [(x, dist2(self.X[x], q)) for c_i in cs for x in self.ivf[c_i]]
        x_dist.sort(key=lambda x_d: x_d[1])
        assert len(x_dist) >= top_k
        return [x_d[0] for x_d in x_dist[:top_k]]

    def batch(self, Q, top_k, **kwargs):
        """Run the Q query_data and return the top_k results for each."""
        return [self.query(q, top_k, **kwargs) for q in Q]
