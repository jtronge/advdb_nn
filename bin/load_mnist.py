"""Test loading script for the MNIST dataset."""
import os
from advdb_nn.loaders import mnist


mnist_dir = os.path.join('datasets', 'MNIST')
label_file = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
img_file = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
for label in mnist.read_labels(label_file):
    print(label)
for img in mnist.read_images(img_file):
    print(img)
