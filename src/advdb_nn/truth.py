"""Truth calculation, dumping and loading."""
import pickle
from advdb_nn.util import dist2


class Truth:
    """Truth class to be dumped to disk."""

    def __init__(self, data, queries, top_k, results):
        """Truth class constructor."""
        self.data = data
        self.queries = queries
        self.top_k = top_k
        self.results = results


def compute_dump_truth(out_f, data, queries, top_k):
    """Compute and dump the truth value """
    results = []
    # Get the k-nn neighbors through brute-force
    for i, q in enumerate(queries):
        print('Computing query', i, 'out of', len(queries))
        dists = [(x_i, dist2(x, q)) for x_i, x in enumerate(data)]
        dists.sort(key=lambda d: d[1])
        results.append([d[0] for d in dists[:top_k]])
    truth = Truth(data, queries, top_k, results)
    with open(out_f, 'wb') as fp:
        pickle.dump(truth, fp)


def load_truth(fname):
    """Load a precalculated truth from a file."""
    with open(fname, 'rb') as fp:
        return pickle.load(fp)
