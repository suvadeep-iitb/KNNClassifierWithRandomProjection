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
from RandomEmbeddingAKNNPredictor import *
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors



class OneVsRestEmbeddingAKNNPredictor(RandomEmbeddingAKNNPredictor):

  def __init__(self, params):
    super(OneVsRestEmbeddingAKNNPredictor, self).__init__(params)


  def LearnParams(self, X, Y, numThreads, itr):
    L = self.labelDim
    D = self.featureDim
    embDim = self.embDim
    C = self.lamb

    # Generate scheudo-random projection matrix
    np.random.seed(self.seed)
    if (embDim == 0):
      R = csr_matrix(identity(L, dtype=np.float32))
      embDim = L
    else:
      R = np.random.randn(L, embDim);

    maxMem = 2**34
    batchSize = int(maxMem/L)
    self.featureProjMatrix = np.zeros((D, embDim), np.float32)
    avgTrainError = 0

    for bs in range(0, L, batchSize):
      be = min(bs + batchSize, L)
      # Perform linear regression using liblinear
      resultList = Parallel(n_jobs = numThreads)(delayed(TrainWrapper)(Y[:, l], X, l, C) for l in range(bs, be))

      avgTrainError += sum([resultList[l][1] for l in range(be-bs)])

      # Collect the model parameters into a matrix
      W = np.zeros((D, 0), dtype=np.float32);
      for l in range(be-bs):
        W = np.hstack((W, resultList[0][0].reshape((-1, 1))))
        del resultList[0]

      if (issparse(R)) :
        self.featureProjMatrix += W * R[bs:be, :]
      else:
        self.featureProjMatrix += np.matmul(W, R[bs:be, :])

    self.trainError = avgTrainError/L
    print("Total training Error: "+str(self.trainError))
  
    '''
    # Put the labels into the queue
    queueLock = threading.Lock()
    labelQueue = myQueue()
    queueLock.acquire()
    for l in range(embDim):
      labelQueue.enqueue(l)
    queueLock.release()

    params = {"X": X, "Z": Z, "C": C, "W": W}

    # Create new threads
    threadList = []
    for tID in range(numThreads):
      thread = myThread(tID, labelQueue, queueLock, TrainWrapper, params)
      thread.start()
      threadList.append(thread)

    # Wait for all threads to complete
    for t in threadList:
      t.join()

    # Return the model parameter
    return params["W"]
    '''


  def MeanSquaredError(self, X, Y, maxSamples):
    raise NotImplementedError


  def CreateAKNNGraph(self, X, numThreads):
    # Get the embedding of X
    pX = self.EmbedFeature(X, numThreads)
    # initialize a new index, using a HNSW index on l2 space
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(pX)
    index.createIndex({'post': 2, 'M': 10, 'maxM0': 20}, print_progress=False)
    '''
    index = NearestNeighbors(n_neighbors = 10, radius = 5, 
                             algorithm = 'auto', metric = 'l2',
                             n_jobs = numThreads)
    index.fit(pX)
    '''
    return index



  def MeanSquaredError(self, X, Y, maxSamples):
    Xsam, Ysam, _ = DownSampleData(X, Y, maxSamples)
    Yemb = Ysam*self.labelProjMatrix
    if (issparse(X)):
      Xemb = Xsam*self.featureProjMatrix
    else:
      Xemb = np.matmul(Xsam, self.featureProjMatrix)
    return mean_squared_error(Yemb, Xemb)



'''
def TrainWrapper(l, params):
  X = params["X"]
  Z = params["Z"][:, l]
  C = params["C"]
  
  model = LinearSVR(epsilon=0.0, 
                    C=C, 
                    loss='squared_epsilon_insensitive', 
                    dual=False, 
                    fit_intercept=False)
  model.fit(X, Z)
  params["W"][:, l] = model.coef_
'''


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






