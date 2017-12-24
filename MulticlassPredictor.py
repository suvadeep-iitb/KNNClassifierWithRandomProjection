import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse, identity
from collections import namedtuple
import pickle
from joblib import Parallel, delayed
import multiprocessing
import nmslib
import math
#from liblinearutil import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVR, LinearSVC
from MyThread import myThread
import threading
from MyQueue import myQueue
from datetime import datetime
from KNNPredictor import *
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors



class MulticlassPredictor:

  def __init__(self, params):
    self.lamb = params['lamb']
    self.featureDim = params['featureDim']
    self.labelDim = params['labelDim']
    self.itr = params['itr']
    self.sampleIndices = []


  def Train(self, X, Y, maxTrainSamples = 0, numThreads = 1):
    assert(X.shape[1] == self.featureDim)
    assert(Y.shape[1] == self.labelDim)

    print(str(datetime.now()) + " : " + "Performing down-sampling")
    # Sample train data for faster training
    if (maxTrainSamples > 0):
      X_sam, Y_sam, samIdx = DownSampleData(X, Y, maxTrainSamples)
      self.maxTrainSamples = X_sam.shape[0]
      self.sampleIndices = samIdx
    else:
      X_sam = X
      Y_sam = Y

    print(str(datetime.now()) + " : " + "Starting training")
    # Perform label projection and learn regression parameters
    self.LearnParams(X_sam, Y_sam, self.itr, numThreads)


  def Predict(self, Xt, numThreads = 1):
    assert(Xt.shape[1] == self.featureDim)

    # Predict labels for input data
    print(str(datetime.now()) + " : " + "Performing prediction")
    predYt, scoreYt = self.ComputeLabelScore(Xt, numThreads)

    return predYt, scoreYt


  def ComputeLabelScore(self, Xt, numThreads = 1):
    nt = Xt.shape[0]
    L = self.labelDim
    batchSize = int(math.ceil(float(nt)/numThreads))
    numBatches = int(math.ceil(float(nt)/batchSize))
    startIdx = [i*batchSize for i in range(numBatches)]
    endIdx = [min((i+1)*batchSize, nt) for i in range(numBatches)]
  
    numCores = numThreads
    resultList = Parallel(n_jobs = numCores)(delayed(self.ComputeLabelScoreInner)(Xt[s: e, :]) for s,e in zip(startIdx, endIdx))
    predYt = vstack([tup[0] for tup in resultList], format='lil')
    scoreYt = vstack([tup[1] for tup in resultList], format='lil')

    assert(predYt.shape[0] == nt)
    assert(scoreYt.shape[0] == nt)
    return predYt, scoreYt


  def ComputeLabelScoreInner(self, Xt):
    if (issparse(Xt)):
      scoreYt = Xt * self.W;
    else:
      scoreYt = np.matmul(Xt, self.W)
    predYt, scoreYt = SortCooMatrix(coo_matrix(scoreYt));
    return predYt, scoreYt


  def PredictAndComputePrecision(self, Xt, Yt, maxTestSamples, numThreads):
    assert(Xt.shape[0] == Yt.shape[0])
    assert(Xt.shape[1] == self.featureDim)
    assert(Yt.shape[1] == self.labelDim)

    # Perform down sampling of input data
    if (maxTestSamples > 0):
      Xt, Yt, testSample = DownSampleData(Xt, Yt, maxTestSamples)

    # Predict labels for input data
    print(str(datetime.now()) + " : " + "Performing prediction")
    predYt, scoreYt = self.Predict(Xt, numThreads)

    # Compute precisions for impute data
    print(str(datetime.now()) + " : " + "Computing precisions")
    precision = self.ComputePrecision(predYt, Yt, 5, numThreads)
    #res = {'Y': Yt, 'predY': predYt, 'scoreY': scoreYt, 'precision': precision, 'testSample': testSample}
    res = {'precision': precision}

    return res


  def ComputePrecision(self, predYt, Yt, K, numThreads):
    assert(Yt.shape[1] == self.labelDim)
    assert(predYt.shape == Yt.shape)

    nt, L = Yt.shape
    batchSize = int(math.ceil(float(nt)/numThreads))
    numBatches = int(math.ceil(float(nt)/batchSize))
    startIdx = [i*batchSize for i in range(numBatches)]
    endIdx = [min((i+1)*batchSize, nt) for i in range(numBatches)]
  
    resultList = Parallel(n_jobs = numThreads)(delayed(ComputePrecisionInner)(predYt[s: e, :], Yt[s: e, :], K) for s,e in zip(startIdx, endIdx))
    precision = np.zeros((K, 1))
    for i, res in enumerate(resultList):
      precision += res * (endIdx[i] - startIdx[i])
    precision /= float(nt)
    return precision


  def EmbedFeature(self, X, numThreads=1):
    if (issparse(X)):
      pX = X * self.W
    else:
      pX = np.matmul(X, self.W);
    return pX


  def GetParamMatrix(self):
    return self.W


  def LearnParams(self, X, Y, itr, numThreads):
    L = self.labelDim
    D = self.featureDim
    C = self.lamb

    # Perform linear regression using liblinear
    resultList = Parallel(n_jobs = numThreads)(delayed(TrainWrapper)(Y[:, l], X, l, C) for l in range(L))

    # Collect the model parameters into a matrix
    W = np.zeros((D, L), dtype=np.float);
    for l in range(L):    
      W[:, l] = resultList[l][0]
    avgTrainError = sum([resultList[l][1] for l in range(L)])/L
    print("Total training Error: "+str(avgTrainError))
 
    self.W = W
    self.trainError = avgTrainError



  def MeanSquaredError(self, X, Y, maxSamples):
    Xsam, Ysam, _ = DownSampleData(X, Y, maxSamples)
    if (issparse(X)):
      Yscore = Xsam*self.W
    else:
      Yscore = np.matmul(Xsam, self.W)
    return mean_squared_error(Ysam, Yscore)



def TrainWrapper(Z, X, l, C):
  print("Starting training for "+str(l)+"th label...")
  '''
  model = LinearSVR(epsilon=0.0,
                    tol=0.000001, 
                    max_iter=5000,
                    C=C, 
                    loss='squared_epsilon_insensitive', 
                    dual=False, 
                    fit_intercept=False)
  '''
  model = LinearSVC(dual=False,
                    C=C,
                    fit_intercept=False)
  model.fit(X, Z.toarray().reshape(-1))
  trainError = mean_squared_error(Z.toarray().reshape(-1), model.predict(X))
  print("Completed training for label: "+str(l)+" . Training error: "+str(trainError))

  return (model.coef_, trainError)


def ComputePrecisionInner(predYt, Yt, K):
  assert(predYt.shape == Yt.shape)
  nt = Yt.shape[0]
  precision = np.zeros((K, 1), dtype=np.float)
  for i in range(Yt.shape[0]):
    nzero = Yt[i, :].getnnz()
    for j in range(min(nzero, K)):
      if (Yt[i, predYt[i, j]] > 0):
        for k in range(j, K):
          precision[k, 0] += 1/float(k+1)
  precision /= float(nt)
  return precision




