import argparse
import os
import sys
import numpy as np
from advdb_nn.loaders import mnist
from advdb_nn.util import dist2, recall
from advdb_nn.ivf import IVF
from advdb_nn.ivfv2 import IVFv2


parser = argparse.ArgumentParser(description='IVF evaluation code')
parser.add_argument('limit', default=30, type=int,
                    help='number of query_data to run')
parser.add_argument('top_k', default=10, type=int,
                    help='top k results to return')
parser.add_argument('c_search', default=2, type=int,
                    help='number of centroids to search')
# TODO: Add in hyperparameters here
args = parser.parse_args()

LIMIT = args.limit
TOP_K = args.top_k
C_SEARCH = args.c_search

mnist_dir = os.path.join('datasets', 'MNIST')
train_file = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
test_file = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
X = np.array([img for img in mnist.read_images(train_file)])
test = np.array([img for img in mnist.read_images(test_file)])

print('Brute force calculation...')
correct = []
for test_q in test[:LIMIT]:
    # Just simply calculate distances between all vectors and sort
    dists = [(x_i, dist2(x, test_q)) for x_i, x in enumerate(X)]
    dists.sort(key=lambda d: d[1])
    correct.append([d[0] for d in dists[:TOP_K]])

print('Building IVF index...')
ivf = IVFv2(X, 10)
print('Running IVF query_data...')
total_recall = 0.0
for test_q_i, test_q in enumerate(test[:LIMIT]):
    print('Query', test_q_i, 'of', LIMIT)
    result = ivf.query(test_q, top_k=TOP_K, c_search=C_SEARCH)
    total_recall += recall(correct[test_q_i], result)
print('Recall@{}:'.format(TOP_K), total_recall / LIMIT)
