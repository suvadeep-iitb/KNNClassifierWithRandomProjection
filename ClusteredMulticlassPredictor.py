from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
import copy
import math
import numpy as np
from ClusteredKNNPredictor import *


class ClusteredMulticlassPredictor(ClusteredKNNPredictor):
  def __init__(self, params):
    self.numClusters = params['numClusters']
    self.clusteringAlgo = params['clusteringAlgo']
    self.embDim = params['embDim']
    self.logFile = params['logFile'] + '.pkl'
    self.maxTestSamples = 0
    self.sampleIndices = []
    self.predictorList = []
    self.clusteringAlgo.update_seed(params['seed'])
    '''
    if (self.logFile):
      newLogFile = params['logFile']+'_clustering_log.pkl'
    else:
      newLogFile = ''
    self.clusteringAlgo.update_log_file(newLogFile)
    '''
    for i in range(self.numClusters):
      newBasePredictor = copy.deepcopy(params['basePredictor'])
      if self.logFile:
        newLogFile = params['logFile']+'_CL'+str(i)+'.pkl'
      else:
        newLogFile = ''
      newSeed = (8191*params['seed'] + i)%(2**16)
      newBasePredictor.UpdateLogFile(newLogFile)
      newBasePredictor.UpdateSeed(newSeed)
      self.predictorList.append(newBasePredictor)



  def Train(self, X, Y, maxTrainSamples, numThreads):
    assert(X.shape[0] == Y.shape[0])
    self.featureDim = X.shape[1]
    self.labelDim = Y.shape[1]
  
    assert(X.shape[1] == self.featureDim)
    assert(Y.shape[1] == self.labelDim)
     
    print(str(datetime.now()) + " : " + "Peforming clustering")
    self.clusteringAlgo.fit(X, Y)
    self.clusterAssignments = self.clusteringAlgo.labels_
    nClusters = int(np.max(self.clusterAssignments) + 1)
    assert(nClusters <= self.numClusters)
    print(str(datetime.now()) + " : " + "Clustering done. # of clusters = "+str(nClusters))
    for cId in range(nClusters):
      print('Cluster '+str(cId)+' : # of samples '+str(np.sum(self.clusterAssignments == cId)))
    self.predictorList = self.predictorList[:nClusters]
    self.fIdMappingList = []
    self.lIdMappingList = []
    for cId in range(nClusters):
      print(str(datetime.now()) + " : " + "Starting training on "+str(cId)+'-th cluster')

      if np.sum(self.clusterAssignments == cId) == 0:
        continue

      Xsam, fIdMapping = CompressDimension(X[np.equal(self.clusterAssignments, cId), :])
      Ysam, lIdMapping = CompressDimension(Y[np.equal(self.clusterAssignments, cId), :])
      print("Cluster "+str(cId)+": # of samples "+str(Xsam.shape[0])+", feature dim "+str(fIdMapping.shape[0])+", label dim "+str(lIdMapping.shape[0]))

      self.fIdMappingList.append(fIdMapping)
      self.lIdMappingList.append(lIdMapping)
      
      self.predictorList[cId].Train(Xsam, Ysam, maxTrainSamples, numThreads)



  def Predict(self, Xt, numThreads = 1):
    assert(Xt.shape[1] == self.featureDim)

    clusterAssignments = self.clusteringAlgo.predict(Xt)
    predYt = lil_matrix((Xt.shape[0], self.labelDim), dtype=int)
    scoreYt = lil_matrix((Xt.shape[0], self.labelDim), dtype=float)
    for i, predictor in enumerate(self.predictorList):
      if np.sum(self.clusterAssignments == i) == 0:
        continue
      Xtsam = Xt[np.equal(clusterAssignments, i), :]
      Xtsam = Xtsam[:, self.fIdMappingList[i]]
      print(str(datetime.now()) + " : " + "Performing prediction on "+str(i)+'-th clusters')
      predYtsam, scoreYtsam = predictor.Predict(Xtsam, numThreads)
      n2oLIdFunc = np.vectorize(lambda x: self.lIdMappingList[i][x])
      predYtsam = n2oLIdFunc(predYtsam)
      predYt[np.equal(clusterAssignments, i), :] = predYtsam
      scoreYt[np.equal(clusterAssignments, i),  :] = scoreYtsam
    return predYt, scoreYt



  def PredictAndComputePrecision(self, Xt, Yt, maxTestSamples = 0, numThreads = 1):
    assert(Xt.shape[0] == Yt.shape[0])
    assert(Xt.shape[1] == self.featureDim)
    assert(Yt.shape[1] == self.labelDim)

    # Perform down sampling of input data
    if (maxTestSamples > 0):
      Xt, Yt, testSample = DownSampleData(Xt, Yt, maxTestSamples)

    clusterAssignments = self.clusteringAlgo.predict(Xt)

    resDict = {}
    for i, predictor in enumerate(self.predictorList):
      if np.sum(self.clusterAssignments == i) == 0:
        continue
      Xtsam = Xt[np.equal(clusterAssignments, i), :]
      Ytsam = Yt[np.equal(clusterAssignments, i), :]
      Xtsam = Xtsam[:, self.fIdMappingList[i]]
      Ytsam = Ytsam[:, self.lIdMappingList[i]]
      print(str(datetime.now()) + " : " + "Computing results on "+str(i)+'-th cluster')
      res = predictor.PredictAndComputePrecision(Xtsam, Ytsam, 0, numThreads)
      resDict[i] = res
      print('Cluster '+str(i)+' : size '+str(sum(clusterAssignments==i))+' prec@1 '+str(res['precision'][0]))
    res = self.CombineResults(resDict, clusterAssignments)
    return res



  def CombineResults(self, resDict, clusterAssignments):
    predY = lil_matrix((clusterAssignments.size, self.labelDim), dtype=int)
    scoreY = lil_matrix((clusterAssignments.size, self.labelDim), dtype=float)
    precision = [0]*5
    for i, res in resDict.items():
      ids = np.equal(clusterAssignments, i)
      precision = [p + np.sum(ids)*q for p,q in zip(precision, res['precision'])]
      if 'predY' in res:
        predY[ids, :] = res['predY']
      if 'scoreY' in res:
        scoreY[ids,  :] = res['scoreY']
    precision = np.array([float(p)/clusterAssignments.size for p in precision])
    combinedRes = {'precision': precision}
    if 'predY' in resDict[0]:
      combinedRes['predY'] = predY
    if 'scoreY' in resDict[0]:
      combinedRes['scoreY'] = scoreY
    return combinedRes
