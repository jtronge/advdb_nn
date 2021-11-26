import os
import sys
import numpy as np
from advdb_nn.loaders import mnist


def dist2(a, b):
    """Squared distance calculation."""
    u = b - a
    return np.dot(u, u)


mnist_dir = os.path.join('datasets', 'MNIST')
train_file = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
test_file = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
X = np.array([img for img in mnist.read_images(train_file)])
test = np.array([img for img in mnist.read_images(test_file)])


class IVF:
    """IVF."""

    def __init__(self, X, c_count):
        """IVF constructor."""
        self.X = X
        # Initially choose random centroids
        centroids = np.random.randint(0, X.shape[0], c_count)

        # Build the inverted index
        self.ivf = []
        for i in range(c_count):
            self.ivf.append([])
        for x_i, x in enumerate(X):
            min_d = None
            min_c_i = None
            for c_i, c_x_i in enumerate(centroids):
                c = X[c_x_i]
                d = dist2(x, c)
                if min_d is None or min_c_i is None or d < min_d:
                    min_d = d
                    min_c_i = c_i
            assert min_c_i is not None
            # Add this dataset vector to the inverted index
            self.ivf[min_c_i].append(x_i)

        # Recalculate the centroids based on the closest dataset vectors
        self.centroids = []
        for c_i, c_ivf in enumerate(self.ivf):
            if c_ivf:
                c = sum(self.X[x_i] for x_i in c_ivf) / len(c_ivf)
            else:
                c = self.X[centroids[c_i]]
            self.centroids.append(c)

    def query(self, q, top_k):
        """Query the top-k vectors closest to q."""
        # Find the closest centroid
        min_d = None
        min_c_i = None
        for c_i, c in enumerate(self.centroids):
            d = dist2(q, c)
            if min_d is None or min_c_i is None or d < min_d:
                min_d = d
                min_c_i = c_i
        assert min_c_i is not None
        # Now compute the distances between q and all those in the inverted
        # index
        x_dist = [(x, dist2(self.X[x], q)) for x in self.ivf[min_c_i]]
        x_dist.sort(key=lambda x_d: x_d[1])
        assert len(x_dist) >= top_k
        return [x_d[0] for x_d in x_dist[:top_k]]


def recall(expected, result):
    """Calculate the recall for a correct and result."""
    exp_s = set(expected)
    res_s = set(result)
    return len(exp_s.intersection(res_s)) / len(exp_s)


LIMIT = 10
TOP_K = 10

print('Brute force...')
correct = []
for test_q in test[:LIMIT]:
    dists = [(x_i, dist2(x, test_q)) for x_i, x in enumerate(X)]
    dists.sort(key=lambda d: d[1])
    correct.append([d[0] for d in dists[:TOP_K]])

print('IVF...')
ivf = IVF(X, 10)
for test_q_i, test_q in enumerate(test[:LIMIT]):
    print('************')
    print('Query', test_q_i)
    print('Correct result:', correct[test_q_i])
    result = ivf.query(test_q, TOP_K)
    print('IVF result:', result)
    print('Recall@{}:'.format(TOP_K), recall(correct[test_q_i], result))
