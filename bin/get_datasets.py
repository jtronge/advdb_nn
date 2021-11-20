#!/usr/bin/env python3
import requests
import os
import gzip


DATASETS_DIR = 'datasets'


def get_file(subdir, url):
    """Download and save a file to `datasets/subdir`."""
    path = os.path.join(DATASETS_DIR, subdir)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    fname = os.path.join(path, os.path.basename(url))
    print('Downloading {}'.format(url))
    with open(fname, 'wb') as fp:
        req = requests.get(url, stream=True)
        for chunk in req.iter_content(1024):
            fp.write(chunk)
    # Decompress the file
    if fname.endswith('.gz'):
        with gzip.open(fname) as gfp:
            with open(os.path.splitext(fname)[0], 'wb') as fp:
                fp.write(gfp.read())

try:
    os.mkdir(DATASETS_DIR)
except FileExistsError:
    pass
# Download all datasets
# MNIST (see http://yann.lecun.com/exdb/mnist/)
get_file('MNIST', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
get_file('MNIST', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
get_file('MNIST', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
get_file('MNIST', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
