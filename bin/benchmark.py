"""Benchmarking code."""
import argparse
import json
import yaml
import os
import time
import numpy as np
from advdb_nn.truth import compute_dump_truth, load_truth
from advdb_nn.loaders import mnist, sift
from advdb_nn.ivf import IVF
from advdb_nn.ivfv2 import IVFv2
from advdb_nn.util import recall


def load_dataset(data_set_type, data_set_file, query_file):
    """Load a data set based on the type."""
    if data_set_type == 'MNIST':
        data = np.array([img for img in mnist.read_images(data_set_file)])
        query_data = np.array([img for img in mnist.read_images(query_file)])
        return data, query_data
    elif data_set_type == 'SIFT':
        data = sift.read_fvecs(data_set_file)
        query_data = sift.read_fvecs(query_file)
        return data, query_data
    else:
        raise RuntimeError('Invalid data set type: {}'.format(data_set_type))


parser = argparse.ArgumentParser(description='IVF benchmarking code')
parser.add_argument('config', help='yaml formatted config file')
parser.add_argument('--brute-force-only', action='store_true', help='only compute brute force')
parser.add_argument('--recompute', action='store_true', help='recompute the database file')
args = parser.parse_args()
# Load the config
with open(args.config) as fp:
    cfg = yaml.load(fp, Loader=yaml.CLoader)
# Now load and run the benchmarks
for benchmark in cfg['benchmarks']:
    # Create database files if they doesn't already exist
    if not os.path.exists(benchmark['database_file']) or args.recompute:
        print('Building brute force index')
        data, query_data = load_dataset(benchmark['data_set_type'],
                                        benchmark['data_set_file'],
                                        benchmark['query_file'])
        print('Total data size:', len(data))
        print('Total query size:', len(query_data))
        print(benchmark['query_limit'])
        compute_dump_truth(benchmark['database_file'],
                           data[:benchmark['data_set_limit']],
                           query_data[:benchmark['query_limit']],
                           top_k=benchmark['top_k'])
    if args.brute_force_only:
        continue
    # Now run the benchmark
    truth = load_truth(benchmark['database_file'])
    # Make sure everything is the expected size
    print(len(truth.query_data))
    assert len(truth.query_data) == benchmark['query_limit']
    assert len(truth.data) == benchmark['data_set_limit']
    title = benchmark['title']
    print('Running benchmark "{}"'.format(title))
    batch_size = benchmark['batch_size']
    index_params = benchmark['index_params']
    params = benchmark['params']
    # Build the index
    start = int(time.time())
    # ivf = IVF(truth.data, 10)
    class_ = IVF if benchmark['version'] == 'IVF' else IVFv2
    ivf = class_(truth.data, **index_params)
    end = int(time.time())
    index_build_time = end - start
    print('Index build time (seconds):', index_build_time)
    # Run each batch
    batches = int(len(truth.query_data) / batch_size)
    print('Running', batches, 'batches in total')
    query_times = []
    recall_rates = []
    for i in range(batches):
        print('----------')
        print('Batch', i)
        # Index of qs data
        x = i * batch_size
        k = (i + 1) * batch_size
        qs = truth.query_data[x:k]
        start = int(time.time())
        result = ivf.batch(qs, top_k=truth.top_k, **params)
        end = int(time.time())
        query_time = end - start
        print('Total time (seconds):', query_time)
        query_times.append(query_time)
        average_recall = sum([recall(truth.results[x + j], res)
                              for j, res in enumerate(result)]) / batch_size
        print('Average recall:', average_recall)
        recall_rates.append(average_recall)
    # Dump the results
    with open(benchmark['output_file'], 'w') as fp:
        json.dump({
            'title': title,
            'index_build_time': index_build_time,
            'query_times': query_times,
            'recall_rates': recall_rates,
        }, fp, indent=4)
