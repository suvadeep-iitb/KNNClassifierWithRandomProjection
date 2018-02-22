import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix as csr
from scipy.spatial import distance
from data_partitioner import DataPartitioner as DP

data = csr(np.random.randn(7, 3), dtype=np.float)
print(str(-distance.cdist(data.todense(), data.todense())))
lm = csr([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1]])
ind = lm.indices
idp = lm.indptr

data_ind = data.indices
data_data = data.data
data_idp = data.indptr

K = 2
num_nn = 3
lan = 1
rf = 1.1
seed = 0
ver = 1

assign = np.zeros((lm.shape[0]), dtype=np.int32)
dp = DP()
dp.RunNeighbourExpansionVP(data_ind, data_data, data_idp, ind, idp, assign, K, num_nn, lan, rf, seed, ver)

print(str(assign))
