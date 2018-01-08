from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
import math
import copy
import numpy as np
from RandomEmbeddingAKNNPredictor import RandomEmbeddingAKNNPredictor

class EnsembleKNNPredictor(RandomEmbeddingAKNNPredictor):

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


  def LearnParams(self, X, Y, itr, numThreads):
    #numThreads = min(numThreads, int(multiprocessing.cpu_count()/2))
    #Parallel(n_jobs = numThreads)(delayed(self.LearnParamsWrapper)(i, X, Y, itr, 1) for i in range(len(self.learnerList)))
    for i in range(len(self.learnerList)):
      self.learnerList[i].LearnParams(X, Y, itr, numThreads)

  def EmbedFeature(self, X, numThreads):
    #numThreads = min(numThreads, int(multiprocessing.cpu_count()/2))
    #pXList = Parallel(n_jobs = numThreads)(delayed(self.GetFeatureEmbeddingWrapper)(i, X) for i in range(len(self.learnerList)))
    pXList = []
    for i in range(len(self.learnerList)):
      pXList.append(self.learnerList[i].EmbedFeature(X))
    return np.hstack(pXList)


  def GetFeatureProjMatrix(self):
    return np.vstack([learner.GetFeatureProjMatrix() for learner in self.learnerList])


  def GetLabelProjMatrix(self):
    return np.vstack([learner.GetLabelProjMatrix() for learner in self.learnerList])


  def LearnParamsWrapper(self, idx, X, Y, itr, numThreads):
    self.learnerList[idx].LearnParams(X, Y, itr, numThreads)


  def GetFeatureEmbeddingWrapper(self, idx, X):
    return self.learnerList[idx].GetFeatureEmbedding(X, 1)
