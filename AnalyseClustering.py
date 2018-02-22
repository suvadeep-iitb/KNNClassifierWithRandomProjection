import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
#from sklearn.cluster import KMeans as kmeans
#from sklearn.cluster import MiniBatchKMeans as kmeans
from MyCluster import LabelNeighbourExpansionEP
import labelCount as lc
from joblib import Parallel, delayed
import multiprocessing



class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt


def LoadData(i):
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  data = pickle.load(open(dataFile, 'rb'))
  data.X = normalize(data.X, norm='l2', axis=1)
  data.Xt = normalize(data.Xt, norm='l2', axis=1)
  return data


def GetClusterLabelStatistic(X, Y, numClusters):
  nLabels = Y.shape[1]
  print("Performing clustering")
  clusters = LabelNeighbourExpansionEP(
                    n_clusters = numClusters,
                    center_file = 'ClusAss/res_wikiLSHTC_centers_S0_NN10_C1.pkl'
                    )
  clusters.fit(X, Y)
  clusterAssignments = clusters.labels_.reshape(-1)
  nClusters = np.max(clusterAssignments) + 1

  clusSizes = np.zeros(nClusters)
  for cId in clusterAssignments:
    clusSizes[cId] += 1

  labelFrequency = np.array(np.sum(Y, axis=0)).reshape(-1)
  sortedIdx = np.argsort(-labelFrequency)
  labelFrequency = labelFrequency[sortedIdx]
  Y = Y[:, sortedIdx]
  clusterLabelStatistic = np.zeros((nLabels, nClusters), dtype=float)
  clusterLabelFreq = np.zeros(nClusters)
  for lId in range(nLabels):
    labelClusAss = clusterAssignments[np.array(Y[:, lId].todense()).reshape(-1) > 0.5]
    for cId in labelClusAss:
      clusterLabelStatistic[lId, cId] += 1.0/float(labelFrequency[lId])
      clusterLabelFreq[cId] += 1
  return clusSizes, labelFrequency, clusterLabelFreq, clusterLabelStatistic


if __name__ == '__main__':
  for dId in [6]:
    data = LoadData(dId)
    numThreads = 1
    for nClus in [35]:
      cs, lf, clf, cls = GetClusterLabelStatistic(data.X, data.Y, nClus)
    resFile = 'clus_statistic_D'+str(dId)+'_C'+str(nClus)+'.pkl'
    pickle.dump((cs, lf, clf, cls), open(resFile, 'wb'))
