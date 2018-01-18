import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix as csr
from data_partitioner import DataPartitioner as DP

lm = csr([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 1]])
ind = lm.indices
idp = lm.indptr

K = 2
num_nn = 2
lan = 1
rf = 1.1
seed = 0
ver = 1

assign = np.zeros((2, lm.shape[0]), dtype=np.float)
dp = DP()
dp.RunNeighbourExpansionEP(ind, idp, assign, K, num_nn, lan, rf, seed, ver)

print(str(assign))
