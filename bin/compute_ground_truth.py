"""Compute the ground truth values for a given dataset."""
import argparse
import os
import numpy as np
from advdb_nn import Dataset
from advdb_nn.loaders import mnist
from advdb_nn.util import dist2, recall
# from advdb_nn.ivf import IVF


parser = argparse.ArgumentParser(description='IVF evaluation code')
parser.add_argument('-o', '--output', help='output file', required=True)
parser.add_argument('-l', '--limit', default=30, type=int,
                    help='number of queries to run')
parser.add_argument('-k', '--top-k', default=10, type=int,
                    help='top k results to return')
# TODO: Add in hyperparameters here
args = parser.parse_args()

OUTPUT = args.output
LIMIT = args.limit
TOP_K = args.top_k

mnist_dir = os.path.join('datasets', 'MNIST')
train_file = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
test_file = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
X = np.array([img for img in mnist.read_images(train_file)])
test = np.array([img for img in mnist.read_images(test_file)])

print('Brute force calculation...')
truth = []
for i, test_q in enumerate(test[:LIMIT]):
    print('Running query', i, 'of', LIMIT)
    # Just simply calculate distances between all vectors and sort
    dists = [(x_i, dist2(x, test_q)) for x_i, x in enumerate(X)]
    dists.sort(key=lambda d: d[1])
    truth.append([d[0] for d in dists[:TOP_K]])

# Create the dataset and dump it
ds = Dataset(X, test[:LIMIT], TOP_K, truth)
ds.dump(OUTPUT)
