import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
from collections import namedtuple
import pickle
from joblib import Parallel, delayed
import multiprocessing
import nmslib
import math
from liblinearutil import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVR
from MyThread import myThread
import threading
from MyQueue import myQueue
from datetime import datetime
from KNNPredictor import *



class RandomEmbeddingAKNNPredictor(KNNPredictor):

  def __init__(self, params):
    self.numLearners = params['numLearners']
    self.embDim = params['embDim']
    self.lamb = params['lamb']
    self.featureDim = params['featureDim']
    self.labelDim = params['labelDim']
    self.maxTrainSamples = 0



  def Train(self, X, Y, maxTrainSamples = 0, numThreads = 1):
    assert(X.shape[1] == self.featureDim)
    assert(Y.shape[1] == self.labelDim)

    print(str(datetime.now()) + " : " + "Performing down-sampling")
    # Sample train data for faster training
    if (maxTrainSamples > 0):
      X_sam, Y_sam = DownSampleData(X, Y, maxTrainSamples)
      self.maxTrainSamples = maxTrainSamples

    print(str(datetime.now()) + " : " + "Starting regression")
    # Perform label projection and learn regression parameters
    self.W = self.LearnRandomEmbeddedRegression(X_sam, Y_sam, numThreads)

    # Create K nearest neighbor graph over training examples
    print(str(datetime.now()) + " : " + "Creating Approximate KNN graph over train examples")
    self.graph = CreateAKNNGraph(self.W, X, numThreads)
    self.Y = Y


 
  def ComputeKNN(self, Xt, nnTest, numThreads = 1):
    # Project the Xt into the embedding space
    if(issparse(Xt)):
      pXt = Xt * self.W
    else:
      pXt = np.matmul(Xt, self.W);

    # get the nearest neighbours for all the test datapoint
    neighbors = self.graph.knnQueryBatch(pXt, nnTest, num_threads=numThreads)
  
    # Create the KNN matrix
    AKNN = np.zeros((pXt.shape[0], nnTest), dtype=np.int64);
    for i,nei in enumerate(neighbors):
      if (len(nei[0]) < nnTest):
        print(str(pXt.shape[0])+'/'+str(i)+' '+str(len(nei[0]))+' '+str(len(nei[1])))
      AKNN[i, :] = nei[0]

    return AKNN



  def LearnRandomEmbeddedRegression(self, X, Y, numThreads):
    L = self.labelDim
    D = self.featureDim
    embDim = self.embDim
    C = self.lamb

    # Generate random projection matrix
    R = np.random.randn(L, embDim);
    R[R > 0] = 1 / math.sqrt(embDim)
    R[R < 0] = -1 / math.sqrt(embDim)

    # Project Y into a lower dimention using random projection matrix
    Z = Y * R;

    # Perform linear regression using liblinear
    resultList = Parallel(n_jobs = numThreads)(delayed(TrainWrapper)(Z[:, l], X, l, C) for l in range(embDim))

    # Collect the model parameters into a matrix
    W = np.zeros((D, embDim), dtype=np.float);
    for l in range(embDim):    
      W[:, l] = resultList[l]
    return W
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
  print("Staring training for "+str(l)+"th label...")
  model = LinearSVR(epsilon=0.0, 
                    C=C, 
                    loss='squared_epsilon_insensitive', 
                    dual=False, 
                    fit_intercept=False)
  model.fit(X, Z)
  print("Completed training for "+str(l)+"th label...")
  return model.coef_



def CreateAKNNGraph(W, X, numThreads):
  # Project the X into the embedding space
  if(issparse(X)):
    pX = X * W
  else:
    pX = np.matmul(X, W)

  # initialize a new index, using a HNSW index on l2 space
  index = nmslib.init(method='hnsw', space='l2')
  index.addDataPointBatch(pX)
  index.createIndex({'post': 2, 'M': 15, 'maxM0': 30}, print_progress=False)

  return index




