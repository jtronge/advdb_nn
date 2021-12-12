#!/bin/sh
TRUTH_FILE=/tmp/sift_ground.pickle
LIMIT=1000
TOP_K=10

#python3 bin/mnist_ground_truth.py -o $TRUTH_FILE \
#	-d datasets/MNIST/train-images-idx3-ubyte \
#	-q datasets/MNIST/t10k-images-idx3-ubyte -l $LIMIT -k $TOP_K
python3 bin/benchmark_ivfv2.py -i $TRUTH_FILE --batch-size 100
