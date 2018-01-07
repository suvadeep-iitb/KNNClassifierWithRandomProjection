import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import labelCount as lc


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


def GetClusterLabelStatistic(X, Y):
  nLabels = Y.shape[1]
  print(str(datetime.now()) + " : " + "Peforming clustering")
  clusters = self.clusteringAlgo(n_clusters = self.numClusters,
                                        n_jobs = numThreads).fit(X)
  clusterAssignments = clusters.labels_
  nClusters = np.max(self.clusterAssignments) + 1

  labelFrequency = np.sum(Y, axis=0).toarray().reshape(-1)
  sortedIdx = np.argsort(-labelFrequency)
  labelFrequency = labelFrequency(sortedIdx)
  Y = Y[:, sortedIdx]
  clusterLabelStatistic = np.zeros((nLabels, nClusters), dtype=float)
  clusterLabelFreq = np.zeros(nClusters)
  for lId in range(nLabels):
    labelClusAss = clusterAssignments[Y[:, lId].toarray().reshape(-1) > 0.5]
    for cId in labelClusAss:
      clusterLabelStatistic[lId, cId] += 1.0/float(labelFrequency[lId])
      clusterLabelFreq[cId] += 1
  return labelFrequency, clusterLabelFreq, clusterLabelStatistic
  
