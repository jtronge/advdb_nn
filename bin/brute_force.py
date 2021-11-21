"""Brute force demo + other experiments."""
import os
import numpy as np
from advdb_nn.loaders import mnist


def debug(*pargs, **kwargs):
    """Print out some debug info."""
    print(*pargs, **kwargs)


def brute_force(X, test, k_count=10):
    """Run a brute force algorithm on the train and test data.

    Based partly on code used for ANN-Benchmarks
    (https://github.com/erikbern/ann-benchmarks)."""
    all_results = []
    for i, q in enumerate(test):
        debug('Brute force query: %i / %i' % (i, len(test)))
        q = np.array(q)
        results = [(i, np.abs(np.dot(x, x) - np.dot(q - x, q - x)))
                   for i, x in enumerate(X)]
        results.sort(key=lambda res: res[1])
        all_results.append([res[0] for res in results[:k_count]])
    return np.array(all_results)


LIMIT = 10

mnist_dir = os.path.join('datasets', 'MNIST')
train_file = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
test_file = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
X = np.array([img for img in mnist.read_images(train_file)])
test = np.array([img for img in mnist.read_images(test_file)])
# print('CALCULATING BRUTE FORCE:')


# TODO: Refactor this into the advdb_nn class and as a class
def quantize(X, m=16, c_count=10):
    """Quantize the training data.

    :param X: data as a numpy array
    :param m: number of distinct subvectors
    :param c_count: number of centroids per subvector
    """
    D = X.shape[1]
    assert (D % m) == 0, 'Data cannot be quantized for the given dimension {}'.format(d)
    # Split the input into subvectors
    U = np.stack([np.stack(np.split(x, m)) for x in X])
    print(U.shape)
    assert U.shape[1] == m
    C = []
    IVF = []
    for i in range(m):
        # Get all the subvectors
        U_i = U[:, i]
        # Generate random centroids
        centroids = np.random.randint(0, U_i.shape[0], c_count)

        def dist2(a, b):
            """Squared distance calculation."""
            u = b - a
            return np.dot(u, u)

        # Calculate the closests centroid for each subvector
        C_i = [min(enumerate(centroids),
                   key=lambda val: dist2(U_i[val[1]], u))[0]
               for u in U_i]
        # Build the inverse list (a map from the centroid index to the list of
        # subvector indices)
        ivf = [[x for x, u in enumerate(U_i) if C_i[x] == j]
               for j in range(c_count)]
        print('IVF')
        # Recalculate centroids
        C_i = [U_i[C_i[j]] for j in range(c_count)]
        for j in range(c_count):
            if ivf[j]:
                C_i[j] = sum(U_i[x] for x in ivf[j]) / len(ivf[j])
                print(C_i[j])
        C.append(C_i)
        IVF.append(ivf)
    return ivf, C


ivf, C = quantize(X)
