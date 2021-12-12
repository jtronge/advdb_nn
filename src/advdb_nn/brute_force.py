"""Brute force implementation."""
from advdb_nn.util import dist2


class BruteForce:
    """Run brute force k-NN to get ground truth values."""

    def __init__(self, data, query_data, top_k):
        """Brute force constructor."""
        self.data = data
        self.query_data = query_data
        self.top_k = top_k
        self.results = []
        # Get the k-nn neighbors through brute-force
        for i, q in enumerate(query_data):
            print('Computing query', i, 'out of', len(query_data))
            dists = [(x_i, dist2(x, q)) for x_i, x in enumerate(data)]
            dists.sort(key=lambda d: d[1])
            self.results.append([d[0] for d in dists[:top_k]])
