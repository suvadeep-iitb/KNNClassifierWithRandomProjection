import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
from MyCluster import LabelNeighbourExpensionEP as ClusAlgo
#from MyCluster import LabelNeighbourExpensionVP as ClusAlgo
#from MyCluster import LabelRand as ClusAlgo
import labelCount as lc


class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt


def MyNormalize(X, Xt, norm):
  print("Normalizing data ...")
  n = X.shape[0]
  if issparse(X):
    XXt = vstack([X, Xt]) 
  else:
    XXt = np.vstack([X, Xt])
  assert(norm in ['l2_row', 'l2_col', 'l1_row', 'l1_col', 'max_row', 'max_col'])
  if 'row' in norm:
    axis = 1
  else:
    axis = 0
  if 'l2' in norm:
    nor = 'l2'
  elif 'l1' in norm:
    nor = 'l1'
  else:
    nor = 'max'
  XXt = normalize(XXt, norm = nor, axis = axis)
  print("Normalization done")
  return XXt[:n, :], XXt[n:, :]


for i in [19]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  resFile = labelStruct.resFile
  print("Running for " + dataFile)
  data = pickle.load(open(dataFile, 'rb'))
  
  # Perform initial random permutation of the data
  '''
  print("Randomly permuting the data ...")
  perm = np.random.permutation(data.X.shape[0])
  data.X = csr_matrix(data.X[perm, :])
  data.Y = csr_matrix(data.Y[perm, :])

  perm = np.random.permutation(data.Xt.shape[0])
  data.Xt = csr_matrix(data.Xt[perm, :])
  data.Yt = csr_matrix(data.Yt[perm, :])
  '''
  
  # Remove label with no sample
  labelCounts = np.array(np.sum(data.Y, axis=0)).reshape(-1)
  nonemptyLabels = (labelCounts > 0)
  data.Y = csr_matrix(data.Y[:, nonemptyLabels])
  data.Yt = csr_matrix(data.Yt[:, nonemptyLabels])

  # Normalize data
  data.X, data.Xt = MyNormalize(data.X, data.Xt, 'l2_row')
 
  for num_nn in [10]:
    for s in range(1, 2):
      for c in [10]:
        print("Running for S = "+str(s)+" C = "+str(c))
        clus_ass_file = ''#'ClusterCenters/NEEP_DeliciousLarge/res_deliciousLarge_clus_ass_'+str(s)+'.pkl'
        nnClus = ClusAlgo(n_clusters = 16,
                          n_init = 1,
                          num_nn = num_nn,
                          rep_factor = 1.2,
                          label_normalize = 1,
                          C = c,
                          max_iter_svc = 50,
                          seed = s,
                          verbose = 1,
                          res_file = resFile+'_clus_ass_S'+str(s)+'_NN'+str(num_nn)+'_C'+str(c)+'.pkl',
                          #clus_ass_file = clus_ass_file,
                          n_jobs = 16)
        nnClus.fit(data.X, data.Y)
        pickle.dump(nnClus.centers_, open(resFile+'_centers_S'+str(s)+'_NN'+str(num_nn)+'_C'+str(c)+'.pkl', 'wb'))
