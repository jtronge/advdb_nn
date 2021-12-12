"""Compute the ground truth for the SIFT dataset."""
import argparse
import pickle
from advdb_nn.loaders import sift
from advdb_nn.truth import compute_dump_truth

parser = argparse.ArgumentParser(description='SIFT ground truth calculation')
parser.add_argument('-o', '--output', help='output file', required=True)
parser.add_argument('-d', '--dataset-file', help='data set file to load', required=True)
parser.add_argument('-q', '--query_data-file', help='query file to load', required=True)
parser.add_argument('-l', '--limit', default=30, type=int,
                    help='number of query_data to run')
parser.add_argument('-k', '--top-k', default=10, type=int,
                    help='top k results to return')
args = parser.parse_args()

# Load the data set and query_data
data = sift.read_fvecs(args.dataset_file)
query_data = sift.read_fvecs(args.query_data_file)
compute_dump_truth(args.output, data, query_data[:args.limit], top_k=args.top_k)
#truth = BruteForce(data, query_data[:args.limit], top_k=args.top_k)
#with open(args.output, 'wb') as fp:
#    pickle.dump(truth, fp)
