from advdb_nn.loaders import sift
from advdb_nn.util import recall
from advdb_nn.ivf import IVF

fname = 'datasets/sift/sift_base.fvecs'
X = data = sift.read_fvecs(fname)
print(fname)
print(data.shape)
fname = 'datasets/sift/sift_groundtruth.ivecs'
truth = data = sift.read_ivecs(fname)
print(fname)
print(data.shape)
fname = 'datasets/sift/sift_learn.fvecs'
data = sift.read_fvecs(fname)
print(fname)
print(data.shape)
fname = 'datasets/sift/sift_query.fvecs'
query = data = sift.read_fvecs(fname)
print(fname)
print(data.shape)

ivf = IVF(X, 10)
# TODO: I think the recall function is incorrect
for q_i, q in enumerate(query):
    result = ivf.query(q, top_k=truth.shape[1], c_search=2)
    print('recall@100', recall(truth[q_i], result))
