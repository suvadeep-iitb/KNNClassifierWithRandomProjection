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
    self.featureDim = params['featureDim']
    self.labelDim = params['labelDim']
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
    assert(X.shape[1] == self.featureDim)
    assert(Y.shape[1] == self.labelDim)
     
    print(str(datetime.now()) + " : " + "Peforming clustering")
    self.clusters = self.clusteringAlgo(n_clusters = self.numClusters,
                                        n_jobs = numThreads).fit(X)
    self.clusterAssignments = self.clusters.labels_
    nClusters = np.max(self.clusterAssignments) + 1
    print(str(datetime.now()) + " : " + "Clustering done. # of clusters = "+str(nClusters))
    self.predictorList = self.predictorList[:nClusters]
    for cId in range(nClusters):
      print(str(datetime.now()) + " : " + "Starting training on "+str(cId)+'-th cluster')
      Xsam = X[np.equal(self.clusterAssignments, cId), :]
      Ysam = Y[np.equal(self.clusterAssignments, cId), :]
      self.predictorList[cId].Train(X, Y, maxTrainSamples, numThreads)



  def Predict(self, Xt, nnTest, numThreads = 1):
    assert(Xt.shape[1] == self.featureDim)

    clusterAssignments = self.clusters.predict(Xt)
    predYt = lil_matrix((Xt.shape[0], self.labelDim), dtype=int)
    scoreYt = lil_matrix((Xt.shape[0], self.labelDim), dtype=float)
    for i, predictor in enumerate(self.predictorList):
      Xtsam = Xt[np.equal(clusterAssignments, i), :]
      print(str(datetime.now()) + " : " + "Performing prediction on "+str(i)+'-th clusters')
      predYtsam, scoreYtsam = predictor.Predict(Xtsam, nnTest, numThreads)
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

