"""Compute the ground truth values for a given dataset."""
import argparse
import numpy as np
from advdb_nn.loaders import mnist
from advdb_nn.truth import compute_dump_truth


parser = argparse.ArgumentParser(description='IVF evaluation code')
parser.add_argument('-o', '--output', help='output file', required=True)
parser.add_argument('-d', '--dataset-file', help='data set file to load', required=True)
parser.add_argument('-q', '--query_data-file', help='query file to load', required=True)
parser.add_argument('-l', '--limit', default=30, type=int,
                    help='number of query_data to run')
parser.add_argument('-k', '--top-k', default=10, type=int,
                    help='top k results to return')
args = parser.parse_args()

#mnist_dir = os.path.join('datasets', 'MNIST')
#train_file = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
#test_file = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
data = np.array([img for img in mnist.read_images(args.dataset_file)])
query_data = np.array([img for img in mnist.read_images(args.query_data_file)])
compute_dump_truth(args.output, data, query_data[:args.limit], top_k=args.top_k)
