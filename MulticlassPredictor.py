import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse, identity
from collections import namedtuple
import pickle
from joblib import Parallel, delayed
import multiprocessing
import math
#from liblinearutil import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVR, LinearSVC
from datetime import datetime
from KNNPredictor import *
from sklearn.metrics import mean_squared_error
#from sklearn.neighbors import NearestNeighbors



class MulticlassPredictor:
  def __init__(self, params):
    self.lamb = params['lamb']
    self.itr = params['itr']
    self.logFile = params['logFile']
    self.seed = params['seed']
    self.embDim = params['embDim']
    self.sampleIndices = []


  def Train(self, X, Y, maxTrainSamples = 0, numThreads = 1):
    assert(X.shape[0] == Y.shape[0])
    self.featureDim = X.shape[1]
    self.labelDim = Y.shape[1]

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
    self.LearnParams(X_sam, Y_sam, numThreads, self.itr)


  def Predict(self, Xt, numThreads = 1):
    assert(Xt.shape[1] == self.featureDim)

    # Predict labels for input data
    print(str(datetime.now()) + " : " + "Performing prediction")
    predYt = self.ComputeLabelScore(Xt, numThreads)

    return predYt


  def UpdateLogFile(self, logFile):
    self.logFile = logFile


  def UpdateSeed(self, seed):
    self.seed = seed


  def LoadModel(self, W):
    self.W = W
    self.featureDim = W.shape[0]-1
    self.labelDim = W.shape[1]


  def ComputeLabelScore(self, Xt, numThreads = 1):
    nt = Xt.shape[0]
    L = self.labelDim
    batchSize = int(math.ceil(float(nt)/numThreads))
    numBatches = int(math.ceil(float(nt)/batchSize))
    startIdx = [i*batchSize for i in range(numBatches)]
    endIdx = [min((i+1)*batchSize, nt) for i in range(numBatches)]
  
    numCores = numThreads
    resultList = Parallel(n_jobs = numCores)(delayed(self.ComputeLabelScoreInner)(Xt[s: e, :]) for s,e in zip(startIdx, endIdx))
    if issparse(resultList[0]):
      predYt = vstack(resultList, format='lil')
    else:
      predYt = np.vstack(resultList)

    assert(predYt.shape[0] == nt)
    return predYt


  def ComputeLabelScoreInner(self, Xt):
    return self.EmbedFeature(Xt)


  def PredictAndComputePrecision(self, Xt, Yt, maxTestSamples = 0, numThreads = 1):
    assert(Xt.shape[0] == Yt.shape[0])
    assert(Xt.shape[1] == self.featureDim)
    assert(Yt.shape[1] == self.labelDim)

    # Perform down sampling of input data
    if (maxTestSamples > 0):
      Xt, Yt, testSample = DownSampleData(Xt, Yt, maxTestSamples)

    maxMem = 2**26
    batchSize = int(maxMem / self.labelDim)
    print(str(datetime.now()) + " : " + "Batch size used for prediction: "+str(batchSize))

    precision = 0
    for bs in range(0, Xt.shape[0], batchSize):
      be = min(bs + batchSize, Xt.shape[0])
      # Predict labels for input data
      print(str(datetime.now()) + " : " + "Performing prediction for "+str(int(bs/batchSize+1))+"-th batch")
      predYt = self.Predict(Xt[bs:be, :], numThreads)

      # Compute precisions for impute data
      print(str(datetime.now()) + " : " + "Computing precisions")
      precBatch = self.ComputePrecision(predYt, Yt[bs:be, :], 5, numThreads)
      precision += precBatch * (be - bs)
    #res = {'Y': Yt, 'predY': predYt, 'scoreY': scoreYt, 'precision': precision, 'testSample': testSample}
    res = {'precision': precision/Xt.shape[0]}

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
    if (X.shape[1] != self.embDim):
      if issparse(X):
        X = X * self.featureProjMatrix
      else:
        X = np.matmul(X, self.featureProjMatrix)
    if (issparse(X)):
      pX = X * self.W[:-1, :] + self.W[-1, :]
    else:
      pX = np.matmul(X, self.W[:-1, :]) + self.W[-1, :];
    return pX


  def GetParamMatrix(self):
    return self.W


  def LearnParams(self, X, Y, numThreads, itr):
    L = self.labelDim
    D = self.featureDim
    C = self.lamb

    self.featureProjMatrix = np.random.randn(D, self.embDim)/np.sqrt(float(self.embDim))
    if issparse(X):
      X = X * self.featureProjMatrix
    else:
      X = np.matmul(X, self.featureProjMatrix)

    # Perform linear regression using liblinear
    resultList = Parallel(n_jobs = numThreads)(delayed(TrainWrapper)(Y[:, l], X, l, C) for l in range(L))

    avgTrainError = sum([resultList[l][2] for l in range(L)])/L
    print("Mean training Error: "+str(avgTrainError))

    # Collect the model parameters into a matrix
    W = np.zeros((self.embDim+1, 0), dtype=np.float);
    for l in range(L):
      coeff = np.vstack((resultList[0][0].reshape((-1, 1)), resultList[0][1].reshape(1, 1)))    
      W = np.hstack((W, coeff))
      del resultList[0]
 
    self.W = W
    self.trainError = avgTrainError



  def MeanSquaredError(self, X, Y, maxSamples):
    if (X.shape != self.embDim):
      if issparse(X):
        X = X * self.featureProjMatrix
      else:
        X = np.matmul(X, self.featureProjMatrix)
      
    Xsam, Ysam, _ = DownSampleData(X, Y, maxSamples)
    Yscore = self.EmbedFeature(Xsam)
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
                    fit_intercept=True)
  model.fit(X, Z.toarray().reshape(-1))
  trainError = mean_squared_error(Z.toarray().reshape(-1), model.predict(X))
  print("Completed training for label: "+str(l)+" . Training error: "+str(trainError))

  return (model.coef_, model.intercept_, trainError)




