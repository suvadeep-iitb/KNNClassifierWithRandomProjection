import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
from collections import namedtuple
import pickle
from joblib import Parallel, delayed
import multiprocessing
import nmslib
import math
#from liblinearutil import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVR
from MyThread import myThread
import threading
from MyQueue import myQueue
from datetime import datetime
from KNNPredictor import *
from sklearn.metrics import mean_squared_error



class RandomEmbeddingAKNNPredictor(KNNPredictor):

  def __init__(self, params):
    self.numLearners = params['numLearners']
    self.embDim = params['embDim']
    self.lamb = params['lamb']
    self.featureDim = params['featureDim']
    self.labelDim = params['labelDim']
    self.maxTrainSamples = 0
    self.trainError = -1
    self.sampleIndices = []



  def Train(self, X, Y, itr = 10, maxTrainSamples = 0, numThreads = 1):
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
    self.LearnParams(X_sam, Y_sam, itr, numThreads)

    # Create K nearest neighbor graph over training examples
    print(str(datetime.now()) + " : " + "Creating Approximate KNN graph over train examples")
    self.graph = CreateAKNNGraph(self.featureProjMatrix, X, numThreads)
    self.Y = Y


 
  def ComputeKNN(self, Xt, nnTest, numThreads = 1):
    # Project the Xt into the embedding space
    if(issparse(Xt)):
      pXt = Xt * self.featureProjMatrix
    else:
      pXt = np.matmul(Xt, self.featureProjMatrix);

    # get the nearest neighbours for all the test datapoint
    neighbors = self.graph.knnQueryBatch(pXt, nnTest, num_threads=numThreads)
  
    # Create the KNN matrix
    AKNN = np.zeros((pXt.shape[0], nnTest), dtype=np.int64);
    for i,nei in enumerate(neighbors):
      if (len(nei[0]) < nnTest):
        print(str(pXt.shape[0])+'/'+str(i)+' '+str(len(nei[0]))+' '+str(len(nei[1])))
      AKNN[i, :] = nei[0]

    return AKNN



  def LearnParams(self, X, Y, itr, numThreads):
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
      W[:, l] = resultList[l][0]
    avgTrainError = sum([resultList[l][1] for l in range(embDim)])/embDim
    print("Total training Error: "+str(avgTrainError))
  
    self.featureProjMatrix = W
    self.labelProjMatrix = R
    self.trainError = avgTrainError
  
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
    Xsam, Ysam, _ = DownSampleData(X, Y, maxSamples)
    Yemb = Ysam*self.labelProjMatrix
    if(issparse(Xsam)):
      Xemb = Xsam * self.featureProjMatrix
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
  print("Staring training for "+str(l)+"th label...")
  model = LinearSVR(epsilon=0.0, 
                    C=C, 
                    loss='squared_epsilon_insensitive', 
                    dual=False, 
                    fit_intercept=False)
  model.fit(X, Z)
  trainError = mean_squared_error(Z, model.predict(X))
  print("Completed training for label: "+str(l)+" . Training error: "+str(trainError))

  return (model.coef_, trainError)



def CreateAKNNGraph(W, X, numThreads):
  # Project the X into the embedding space
  if(issparse(X)):
    pX = X * W
  else:
    pX = np.matmul(X, W)

  # initialize a new index, using a HNSW index on l2 space
  index = nmslib.init(method='hnsw', space='l2')
  index.addDataPointBatch(pX)
  index.createIndex({'post': 2, 'M': 10, 'maxM0': 20}, print_progress=False)

  return index




