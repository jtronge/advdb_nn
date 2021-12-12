"""Truth calculation, dumping and loading."""
import pickle
import time
from advdb_nn.util import dist2


class Truth:
    """Truth class to be dumped to disk."""

    def __init__(self, data, query_data, top_k, results, batch_size,
                 times_per_batch):
        """Truth class constructor."""
        self.data = data
        self.query_data = query_data
        self.top_k = top_k
        self.results = results
        self.batch_size = batch_size
        self.times_per_batch = times_per_batch


def compute_dump_truth(out_f, data, query_data, top_k, batch_size):
    """Compute and dump the truth value """
    results = []
    times_per_batch = []
    last_time = time.time()
    # Get the k-nn neighbors through brute-force
    for i, q in enumerate(query_data):
        print('Computing query', i, 'out of', len(query_data))
        dists = [(x_i, dist2(x, q)) for x_i, x in enumerate(data)]
        dists.sort(key=lambda d: d[1])
        results.append([d[0] for d in dists[:top_k]])
        # Record batch calculation times
        if (i % batch_size) == (batch_size - 1):
            t = time.time()
            times_per_batch.append(t - last_time)
            print('Batch time of', times_per_batch[-1], 'seconds')
            last_time = t
    truth = Truth(data, query_data, top_k, results, batch_size, times_per_batch)
    # Dump the pickle file
    with open(out_f, 'wb') as fp:
        pickle.dump(truth, fp)


def load_truth(fname):
    """Load a precalculated truth from a file."""
    with open(fname, 'rb') as fp:
        return pickle.load(fp)
