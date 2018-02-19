from joblib import Parallel, delayed
import multiprocessing
import nmslib
from datetime import datetime
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

class KNNPredictor:

  def __init__(self, params):
    self.logFile = params['logFile']
    self.seed = params['seed']



  def Train(self, X, Y, maxTrainSamples, numThreads):
    assert(X.shape[0] == Y.shape[0])

    self.X = X 
    self.Y = Y



  def Predict(self, Xt, nnTest, numThreads = 1):
    # Compute K nearest neighbors for input data
    print(str(datetime.now()) + " : " + "Computing Approximate KNN")
    knn = self.ComputeKNN(Xt, nnTest, numThreads);
    
    # Predict labels for input data
    print(str(datetime.now()) + " : " + "Performing prediction")
    predYt = self.ComputeLabelScore(knn, nnTest, numThreads)

    return predYt



  def ComputeLabelScore(self, KNN, nnTest, numThreads = 1):
    Y = self.Y
    nt = KNN.shape[0]
    L = Y.shape[1]
    batchSize = int(math.ceil(float(nt)/numThreads))
    numBatches = int(math.ceil(float(nt)/batchSize))
    startIdx = [i*batchSize for i in range(numBatches)]
    endIdx = [min((i+1)*batchSize, nt) for i in range(numBatches)]
  
    numCores = numThreads
    resultList = Parallel(n_jobs = numCores)(delayed(ComputeLabelScoreInner)(Y, KNN[s: e, :], nnTest) for s,e in zip(startIdx, endIdx))
    predYt = vstack(resultList, format='lil')

    assert(predYt.shape[0] == nt)
    return predYt



  def ComputePrecision(self, predYt, Yt, K, numThreads):
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



  def ComputeKNN(self, Xt, nnTest, numThreads = 1):
    dist = euclidean_distances(Xt, self.X)
    knn = np.argsort(dist, axis = 1)[:, :nnTest]
    return knn
  


  def PredictAndComputePrecision(self, Xt, Yt, nnTestList, maxTestSamples, numThreads):
    assert(Xt.shape[0] == Yt.shape[0])

    # Perform down sampling of input data
    if (maxTestSamples > 0):
      Xt, Yt, testSample = DownSampleData(Xt, Yt, maxTestSamples)

    maxNNTest = max(nnTestList)
    # Compute K nearest neighbors for input data
    print(str(datetime.now()) + " : " + "Computing KNN")
    print(str(nnTestList))
    knn = self.ComputeKNN(Xt, maxNNTest, numThreads);
    
    resList = []
    for nnTest in nnTestList:
      # Predict labels for input data
      print(str(datetime.now()) + " : " + "Performing prediction for nnTest = " + str(nnTest))
      predYt = self.ComputeLabelScore(knn, nnTest, numThreads)

      # Compute precisions for input data
      print(str(datetime.now()) + " : " + "Computing precisions for nnTest = " + str(nnTest))
      precision = self.ComputePrecision(predYt, Yt, 5, numThreads)
      #resList.append({'Y': Yt, 'predY': predYt, 'scoreY': scoreYt, 'precision': precision, 'testSample': testSample})
      resList.append({'precision': precision})

    return resList



  def UpdateLogFile(self, logFile):
    self.logFile = logFile



  def UpdateSeed(self, seed):
    self.seed = seed



def ComputeLabelScoreInner(Y, KNN, nnTest):
  assert(KNN.shape[1] >= nnTest)
  KNN = KNN[:, :nnTest]
  nt = KNN.shape[0]
  L = Y.shape[1]
  predYt = lil_matrix((nt, L));
  for i in range(nt):
    predYt[i, :] = np.mean(Y[KNN[i, :], :], axis = 0)
  return predYt



def ComputePrecisionInner(predYt, Yt, K):
  assert(predYt.shape == Yt.shape)
  sortedYt, _ = SortCooMatrix(coo_matrix(predYt))
  nt = Yt.shape[0]
  precision = np.zeros((K, 1), dtype=np.float)
  for i in range(Yt.shape[0]):
    nzero = Yt[i, :].getnnz()
    for j in range(min(nzero, K)):
      if (Yt[i, sortedYt[i, j]] > 0):
        for k in range(j, K):
          precision[k, 0] += 1/float(k+1)
  precision /= float(nt)
  return precision



def DownSampleData(X, Y, sampleSize):
  n = X.shape[0]
  if (n > sampleSize):
    perm = np.random.permutation(n)[:sampleSize]
    Xnew = X[perm, :]
    Ynew = Y[perm, :]
  else:
    Xnew = X
    Ynew = Y
    perm = []
  return Xnew, Ynew, perm



def SortCooMatrix(M):
  tuples = zip(M.row, M.col, -M.data);
  sortedTuples = sorted(tuples, key=lambda x: (x[0], x[2]))

  # Create a sparse matrix 
  sortedIdx = lil_matrix(M.shape, dtype=np.uint64);
  sortedVal = lil_matrix(M.shape, dtype=np.float);
  colIdx = 0
  rowIdx = 0
  for t in sortedTuples:
    if t[0] == rowIdx:
      sortedIdx[rowIdx, colIdx] = t[1]
      sortedVal[rowIdx, colIdx] = -t[2]
      colIdx += 1
    elif (t[0] > rowIdx):
      rowIdx = t[0]
      sortedIdx[rowIdx, 0] = t[1]
      sortedVal[rowIdx, 0] = -t[2]
      colIdx = 1
    else:
      assert(False)
  return sortedIdx, sortedVal 

