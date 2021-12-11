"""Run benchmarks on the query."""
import argparse
import time
from advdb_nn.truth import load_truth
from advdb_nn.ivf import IVF


parser = argparse.ArgumentParser(description='main benchmarking script')
parser.add_argument('-i', '--input', help='input truth file', required=True)
parser.add_argument('--c-search', default=1, type=int, help='number of centroids to search')
parser.add_argument('--batch-size', default=30, type=int, help='batch size to run')
args = parser.parse_args()

truth = load_truth(args.input)
ivf = IVF(truth.data, 10)
batches = int(len(truth.queries) / args.batch_size)
for i in range(batches):
    qs = truth.queries[i * args.batch_size:(i + 1) * args.batch_size]
    start = int(time.time())
    ivf.batch(qs, truth.top_k, c_search=args.c_search)
    end = int(time.time())
    print('Total time (seconds):', end - start)
