import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse, identity
import pickle
from joblib import Parallel, delayed
import multiprocessing
import nmslib
import math
#from liblinearutil import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVR, LinearSVC
from datetime import datetime
from AKNNPredictor import *
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors



class RandomProjAKNNPredictor(AKNNPredictor):

  def __init__(self, params):
    self.embDim = params['embDim']
    self.lamb = params['lamb']
    self.maxTrainSamples = 0
    self.seed = params['seed']
    self.logFile = params['logFile']
    self.trainError = -1
    self.sampleIndices = []


  def Train(self, X, Y, maxTrainSamples = 0, numThreads = 1, itr = 10):
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
    self.LearnParams(X_sam, Y_sam, numThreads, itr)

    # Create K nearest neighbor graph over training examples
    print(str(datetime.now()) + " : " + "Creating Approximate KNN graph over train examples")
    self.graph = self.CreateAKNNGraph(X, max(numThreads, 30))
    self.Y = Y


 
  def ComputeAKNN(self, Xt, nnTest, numThreads = 1):
    # Get the embedding of Xt 
    pXt = self.EmbedFeature(Xt, 1)
    # get the nearest neighbours for all the test datapoint
    neighbors = self.graph.knnQueryBatch(pXt, nnTest, num_threads=numThreads)
    # Create the KNN matrix
    AKNN = np.zeros((pXt.shape[0], nnTest), dtype=np.int64);
    for i,nei in enumerate(neighbors):
      if (len(nei[0]) < nnTest):
        print(str(pXt.shape[0])+'/'+str(i)+' '+str(len(nei[0]))+' '+str(len(nei[1])))
      AKNN[i, :] = nei[0]
    return AKNN


  def EmbedFeature(self, X, numThreads=1):
    if (issparse(X)):
      pX = X * self.featureProjMatrix
    else:
      pX = np.matmul(X, self.featureProjMatrix)
    return pX


  def GetFeatureProjMatrix(self):
    return self.featureProjMatrix


  def GetLabelProjMatrix(self):
    return self.labelProjMatrix


  def LearnParams(self, X, Y, numThreads, itr):
    L = Y.shape[1]
    D = X.shape[1]
    embDim = self.embDim
    C = self.lamb

    # Generate scheudo-random projection matrix
    np.random.seed(self.seed)
    R = np.random.randn(L, embDim);
    R[R > 0] = 1 / math.sqrt(embDim)
    R[R < 0] = -1 / math.sqrt(embDim)

    # Project Y into a lower dimention using random projection matrix
    Z = Y * R
    del R

    # Perform linear regression using liblinear
    resultList = Parallel(n_jobs = numThreads)(delayed(TrainWrapper)(Z[:, l], X, l, C) for l in range(embDim))

    avgTrainError = sum([resultList[l][1] for l in range(embDim)])/embDim
    print("Mean training Error: "+str(avgTrainError))

    # Collect the model parameters into a matrix
    W = np.reshape(resultList[0][0], (-1, 1))
    del resultList[0]
    for l in range(embDim-1):
      W = np.hstack((W, np.reshape(resultList[0][0], (-1, 1))))
      del resultList[0]
    self.featureProjMatrix = W
    #self.labelProjMatrix = R
    self.trainError = avgTrainError
  

  def CreateAKNNGraph(self, X, numThreads):
    # Get the embedding of X
    pX = self.EmbedFeature(X, numThreads)
    # initialize a new index, using a HNSW index on l2 space
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(pX)
    index.createIndex({'post': 2, 'M': 10, 'maxM0': 20}, print_progress=False)
    return index


  def MeanSquaredError(self, X, Y, maxSamples):
    Xsam, Ysam, _ = DownSampleData(X, Y, maxSamples)
    Yemb = Ysam*self.labelProjMatrix
    if (issparse(X)):
      Xemb = Xsam*self.featureProjMatrix
    else:
      Xemb = np.matmul(Xsam, self.featureProjMatrix)
    return mean_squared_error(Yemb, Xemb)



def TrainWrapper(Z, X, l, C):
  print("Starting training for "+str(l)+"th label...")
  model = LinearSVR(epsilon=0.0,
                    tol=0.000001, 
                    max_iter=5000,
                    C=C, 
                    loss='squared_epsilon_insensitive', 
                    dual=False, 
                    fit_intercept=True)
  model.fit(X, Z.reshape(-1))
  trainError = mean_squared_error(Z.reshape(-1), model.predict(X))
  print("Completed training for label: "+str(l)+" . Training error: "+str(trainError))

  return (model.coef_, trainError)






