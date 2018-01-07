from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
import math
import numpy as np
from KNNPredictor import *


class ClusteredKNNPredictor(KNNPredictor):
  def __init__(self, params):
    self.numClusters = params['numClusters']
    self.basePredictor = params['basePredictor']
    self.clusteringAlgo = params['clusteringAlgo']
    self.embDim = params['embDim']
    self.maxTestSamples = 0
    self.sampleIndices = []
    self.predictorList = []
    for i in range(self.numClusters):
      newParams = params.copy()
      newParams['logFile'] += '_CL'+str(i)+'.pkl'
      newParams['seed'] = (123*params['seed'] + i)%self.numClusters
      self.predictorList.append(self.basePredictor(newParams))



  def Train(self, X, Y, maxTrainSamples, numThreads):
    assert(X.shape[0] == Y.shape[0])
    self.featureDim = X.shape[1]
    self.labelDim = Y.shape[1]
  
    assert(X.shape[1] == self.featureDim)
    assert(Y.shape[1] == self.labelDim)
     
    print(str(datetime.now()) + " : " + "Peforming clustering")
    print(str(datetime.now()) + " : " + "Performing clustering")
    self.clusters = self.clusteringAlgo(n_clusters = self.numClusters,
                                        max_iter = 10,
                                        n_init = 5,
                                        n_jobs = 5).fit(X)
    self.clusterAssignments = self.clusters.labels_
    nClusters = np.max(self.clusterAssignments) + 1
    print(str(datetime.now()) + " : " + "Clustering done. # of clusters = "+str(nClusters))
    self.predictorList = self.predictorList[:nClusters]
    self.fIdMappingList = []
    self.lIdMappingList = []
    for cId in range(nClusters):
      print(str(datetime.now()) + " : " + "Starting training on "+str(cId)+'-th cluster')
      Xsam, fIdMapping = CompressDimension(X[np.equal(self.clusterAssignments, cId), :])
      Ysam, lIdMapping = CompressDimension(Y[np.equal(self.clusterAssignments, cId), :])

      self.fIdMappingList.append(fIdMapping)
      self.lIdMappingList.append(lIdMapping)
      
      self.predictorList[cId].Train(Xsam, Ysam, maxTrainSamples, numThreads)



  def Predict(self, Xt, nnTest, numThreads = 1):
    assert(Xt.shape[1] == self.featureDim)

    clusterAssignments = self.clusters.predict(Xt)
    predYt = lil_matrix((Xt.shape[0], self.labelDim), dtype=int)
    scoreYt = lil_matrix((Xt.shape[0], self.labelDim), dtype=float)
    for i, predictor in enumerate(self.predictorList):
      Xtsam = Xt[np.equal(clusterAssignments, i), :]
      Xtsam = Xtsam[:, self.fIdMappingList[i]]
      print(str(datetime.now()) + " : " + "Performing prediction on "+str(i)+'-th clusters')
      predYtsam, scoreYtsam = predictor.Predict(Xtsam, nnTest, numThreads)
      n2oLIdFunc = np.vectorize(lambda x: self.lIdMappingList[i][x])
      predYtsam = n2oLIdFunc(predYtsam)
      predYt[np.equal(clusterAssignments, i), :] = predYtsam
      scoreYt[np.equal(clusterAssignments, i),  :] = scoreYtsam
    return predYt, scoreYt



  def PredictAndComputePrecision(self, Xt, Yt, nnTestList, maxTestSamples, numThreads):
    assert(Xt.shape[0] == Yt.shape[0])
    assert(Xt.shape[1] == self.featureDim)
    assert(Yt.shape[1] == self.labelDim)

    # Perform down sampling of input data
    if (maxTestSamples > 0):
      Xt, Yt, testSample = DownSampleData(Xt, Yt, maxTestSamples)

    clusterAssignments = self.clusters.predict(Xt)
    resList = []
    for i, predictor in enumerate(self.predictorList):
      Xtsam = Xt[np.equal(clusterAssignments, i), :]
      Ytsam = Yt[np.equal(clusterAssignments, i), :]
      Xtsam = Xtsam[:, self.fIdMappingList[i]]
      Ytsam = Ytsam[:, self.lIdMappingList[i]]
      print(str(datetime.now()) + " : " + "Computing results on "+str(i)+'-th cluster')
      res = predictor.PredictAndComputePrecision(Xtsam, Ytsam, nnTestList, 0, numThreads)
      resList.append(res)
    res = self.CombineResults(resList, clusterAssignments, nnTestList)
    return res



  def CombineResults(self, resList, clusterAssignments, nnTestList):
    combinedResList = []
    for nn in range(len(nnTestList)):
      predY = lil_matrix((clusterAssignments.size, self.labelDim), dtype=int)
      scoreY = lil_matrix((clusterAssignments.size, self.labelDim), dtype=float)
      precision = [0]*5
      for i, res in enumerate(resList):
        ids = np.equal(clusterAssignments, i)
        precision = [p + np.sum(ids)*q for p,q in zip(precision, res[nn]['precision'])]
        if 'predY' in res[nn]:
          predY[ids, :] = res[nn]['predY']
        if 'scoreY' in res[nn]:
          scoreY[ids,  :] = res[nn]['scoreY']
      precision = [float(p)/clusterAssignments.size for p in precision]
      combinedRes = {'precision': precision}
      if 'predY' in resList[0][nn]:
        combinedRes['predY'] = predY
      if 'scoreY' in resList[0][nn]:
        combinedRes['scoreY'] = scoreY
      combinedResList.append(combinedRes)
    return combinedResList


def CompressDimension(X):
  colSum = np.sum(np.abs(X), axis=0)
  _, idMapping = np.where(colSum > 0.0)
  X = X[:, idMapping]
  return X, idMapping
