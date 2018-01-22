from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
import math
import copy
import numpy as np
from KNNPredictor import *

class EnsembleKNNPredictor(KNNPredictor):
  def __init__(self, params):
    self.numLearners = params['numLearners']
    self.basePredictor = params['basePredictor']
    self.embDim = params['embDim'] * self.numLearners
    self.logFile = params['logFile']
    self.maxTestSamples = 0
    self.sampleIndices = []
    self.learnerList = []
    for i in range(self.numLearners):
      newBasePredictor = copy.deepcopy(params['basePredictor'])
      if self.logFile:
        newLogFile = params['logFile']+'_LR'+str(i)+'.pkl'
      else:
        newLogFile = ''
      newSeed = (8191*params['seed'] + i)%(2**16)
      newBasePredictor.UpdateLogFile(newLogFile)
      newBasePredictor.UpdateSeed(newSeed)
      self.learnerList.append(newBasePredictor)


  def Train(self, X, Y, numThreads = 1, itr = 10):
    for i in range(self.numLearners):
      print(str(datetime.now()) + " : " + "Performing training for " + str(i) + "-th learner")
      self.learnerList[i].Train(X, Y, numThreads = numThreads, itr = itr)


  def EmbedFeature(self, X, numThreads):
    raise NotImplemented


  def Predict(self, Xt, nnTest, numThreads = 1):
    for i in range(self.numLearners):
      print(str(datetime.now()) + " : " + "Performing prediction for " + str(i) + "-th learner")
      predYt = self.learnerList[i].Predict(Xt, nnTest, numThreads)
      if i == 0:
        predYtCum = predYt
      else:
        predYtCum += predYt
    predYtCum /= float(self.numLearners)


  def ComputeLabelScore(self, KNN, nnTest, numThreads = 1):
    raise NotImplemented


  def ComputeKNN(self, Xt, nnTest, numThreads = 1):
    raise NotImplemented


  def PredictAndComputePrecision(self, Xt, Yt, nnTestList, maxTestSamples, numThreads):
    assert(Xt.shape[0] == Yt.shape[0])

    # Perform down sampling of input data
    if (maxTestSamples > 0):
      Xt, Yt, testSample = DownSampleData(Xt, Yt, maxTestSamples)

    maxNNTest = max(nnTestList)
    predYtList = []
    for nn in range(len(nnTestList)):
      predYtList.append(lil_matrix(Yt.shape, dtype=np.float))
    for i in range(self.numLearners):
      # Compute K nearest neighbors for i-th learner
      print(str(datetime.now()) + " : " + "Computing KNN for " + str(i) + "-th learner")
      knn = self.learnerList[i].ComputeKNN(Xt, maxNNTest, numThreads)

      for nn, nnTest in enumerate(nnTestList):
        # predict labels
        print(str(datetime.now()) + " : " + "Performing prediction for " + str(i) + "-th learner and nnTest = " + str(nnTest))
        predYtList[nn] += self.learnerList[i].ComputeLabelScore(knn, nnTest, numThreads)

    for nn in range(len(nnTestList)):
      predYtList[nn] /= self.numLearners

    resList = []
    for nn in range(len(nnTestList)):
      # Compute precisions
      print(str(datetime.now()) + " : " + "Computing precisions for nnTest = " + str(nnTestList[nn]))
      precision = self.ComputePrecision(predYtList[nn], Yt, 5, numThreads)
      #resList.append({'Y': Yt, 'predY': predYt, 'scoreY': scoreYt, 'precision': precision, 'testSample': testSample})
      resList.append({'precision': precision})

    return resList

