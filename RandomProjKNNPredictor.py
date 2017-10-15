import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, hstack, vstack
from collections import namedtuple
import pickle
from joblib import Parallel, delayed
import multiprocessing
import nmslib
import math
from liblinearutil import *
#from itertools import zip

def RandProj(X, Y, params):
  L = Y.shape[1]
  D = X.shape[1]
  embDim = params["embDim"]
  numThreads = params["numThreads"]
  C = params["lamb"]

  # Generate projection matrix
  R = np.random.randn(L, embDim);
  R[R > 0] = 1 / math.sqrt(embDim)
  R[R < 0] = -1 / math.sqrt(embDim)

  # Project Y into a lower dimention
  Z = Y * R;

  # Perform linear regression using liblinear
  W = np.zeros((D, embDim), dtype=np.float);
  libArgs = '-s 11 -p 0 -c '+str(C)+' -q'
  numCores = numThreads; # multiprocessing.cpu_count()
  resultList = Parallel(n_jobs = numCores)(delayed(TrainWrapper)(Z[:, l], X, libArgs) for l in range(embDim))
  '''
  for l in range(embDim):
    model = train(Z[:, l], X, libArgs);
    W[:, l] = model.get_decfun()[0]
  '''

  print(str(len(resultList[0])))
  # Collect the model parameters into a matrix
  for l in range(embDim):    
    W[:, l] = resultList[l]

  # Return the model parameter
  return W



def TrainWrapper(Z, X, libArgs):
  model = train(Z, X, libArgs)
  return model.get_decfun()[0]




def ComputeKNN(W, X, Xt, nnTest, numThreads):
  # Project the X and Xt into the embedding space
  pX = X * W;
  pXt = Xt * W;

  # initialize a new index, using a HNSW index on l2 space
  index = nmslib.init(method='hnsw', space='l2')
  index.addDataPointBatch(pX)
  index.createIndex({'post': 2, 'M':25, 'maxM':25, 'maxM0':50, 'delaunay_type':1}, print_progress=False)

  # get the nearest neighbours for all the test datapoint
  neighbors = index.knnQueryBatch(pXt, nnTest, num_threads=numThreads)
  
  # Create the KNN matrix
  KNN = np.zeros((pXt.shape[0], nnTest), dtype=np.int64);
  for i,nei in enumerate(neighbors):
    if (len(nei[0]) < nnTest):
      print(str(pXt.shape[0])+'/'+str(i)+' '+str(len(nei[0]))+' '+str(len(nei[1])))
    KNN[i, :] = nei[0]

  return KNN


def SortCooMatrix(M):
  tuples = zip(M.row, M.col, M.data);
  sortedTuples = sorted(tuples, key=lambda x: (x[0], x[2]))

  # Create a sparse matrix 
  sortedIdx = lil_matrix(M.shape, dtype=np.uint64);
  sortedVal = lil_matrix(M.shape, dtype=np.float);
  colIdx = 0
  rowIdx = 0
  for t in sortedTuples:
    if t[0] == rowIdx:
      sortedIdx[rowIdx, colIdx] = t[1]
      sortedVal[rowIdx, colIdx] = t[2]
      colIdx += 1
    elif (t[0] == rowIdx+1):
      rowIdx += 1
      sortedIdx[rowIdx, 0] = t[1]
      sortedVal[rowIdx, 0] = t[2]
      colIdx = 1
    else:
      assert(false)
  return sortedIdx, sortedVal 
      
    


def PredictY(Y, KNN, nnTest):
  assert(KNN.shape[1] >= nnTest)
  KNN = KNN[:, :nnTest]
  nt = KNN.shape[0]
  L = Y.shape[1]
  scoreYt = lil_matrix((nt, L));
  for i in range(nt):
    scoreYt[i, :] = np.mean(Y[KNN[i, :], :], axis = 0)
  predYt, scoreYt = SortCooMatrix(coo_matrix(scoreYt));
  return predYt, scoreYt



def PredictYParallel(Y, KNN, nnTest, numThreads):
  nt = KNN.shape[0]
  L = Y.shape[1]
  batchSize = int(math.ceil(float(nt)/numThreads))
  startIdx = [i*batchSize for i in range(numThreads)]
  endIdx = [min((i+1)*batchSize, nt) for i in range(numThreads)]
  
  numCores = numThreads;
  resultList = Parallel(n_jobs = numCores)(delayed(PredictY)(Y, KNN[s: e, :], nnTest) for s,e in zip(startIdx, endIdx))
  predYt = vstack([tup[0] for tup in resultList], format='lil')
  scoreYt = vstack([tup[1] for tup in resultList], format='lil')
  assert(predYt.shape[0] == nt)
  assert(scoreYt.shape[0] == nt)
  return predYt, scoreYt



 
def ComputePrecisionParallel(predYt, Yt, K, numThreads):
  assert(predYt.shape == Yt.shape)
  nt, L = Yt.shape
  batchSize = int(math.ceil(float(nt)/numThreads))
  startIdx = [i*batchSize for i in range(numThreads)]
  endIdx = [min((i+1)*batchSize, nt) for i in range(numThreads)]
  
  numCores = numThreads;
  resultList = Parallel(n_jobs = numCores)(delayed(ComputePrecision)(predYt[s: e, :], Yt[s: e, :], K) for s,e in zip(startIdx, endIdx))
  precision = np.zeros((K, 1))
  for i, res in enumerate(resultList):
    precision += res * (endIdx[i] - startIdx[i])
  precision /= float(nt)
  return precision




def ComputePrecision(predYt, Yt, K):
  assert(predYt.shape == Yt.shape)
  nt = Yt.shape[0]
  precision = np.zeros((K, 1), dtype=np.float)
  for i in range(Yt.shape[0]):
    for j in range(K):
      if (Yt[i, predYt[i, j]] > 0.5):
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
  return Xnew, Ynew



def RandomProjKNNPredictor(X, Y, Xt, Yt, params, nnTestList):
  # Make sure label index is stating from 1
  if ((Y[:, 0].nnz != 0) or (Yt[:, 0].nnz != 0)):
    Y = csr_matrix(hstack((csr_matrix((Y.shape[0], 1)), Y)))
    Yt = csr_matrix(hstack((csr_matrix((Yt.shape[0], 1)), Yt)))

  maxTestSamples = params["maxTestSamples"]
  maxTrainSamples= params["maxTrainSamples"]
  # Sample test data for faster compuation of test precision
  Xt, Yt = DownSampleData(Xt, Yt, maxTestSamples)
  # Sample train data for faster training
  X_sam, Y_sam = DownSampleData(X, Y, maxTrainSamples)
  # sample train data for faster computation of train precision
  X_sam_t, Y_sam_t = DownSampleData(X, Y, maxTestSamples)

  W = RandProj(X_sam, Y_sam, params);

  print("\tTraining Finished")

  maxNNTest = max(nnTestList);
  numThreads = params["numThreads"];

  # Compute K nearest neighbors for sampled test examples
  print("\tComputing KNN of test examples")
  KNN = ComputeKNN(W, X, Xt, maxNNTest, numThreads);

  # Compute K nearest neighbors for sampled train examples
  print("\tComputing KNN of training examples")
  KNN_tr = ComputeKNN(W, X, X_sam_t, maxNNTest, numThreads);

  for nnt in range(len(nnTestList)):
    nnTest = nnTestList[nnt]

    # Predict labels for sampled test data
    predYt, scoreYt = PredictYParallel(Y, KNN, nnTest, numThreads)

    # Compute precisions for sampled test data
    precision = ComputePrecisionParallel(predYt, Yt, 5, numThreads)

    # Predict labels for sampled train data
    predYt_tr, scoreYt_tr = PredictYParallel(Y, KNN_tr, nnTest, numThreads)

    # Compute precisions for sampled train data
    precision_tr = ComputePrecisionParallel(predYt_tr, Y_sam_t, 5, numThreads)

    # Save result
    res = {}
    res["precision"] = precision
    res["predictedLabel"] = predYt
    res["score"] = scoreYt
    res["precision_tr"] = precision_tr
    res["predictedLabel_tr"] = predYt_tr
    res["score_tr"] = scoreYt_tr

    resFile = 'Results/'+params["resFilePrefix"]+'_L'+str(params["lamb"])+'_D'+str(params["embDim"])+'_NN'+str(nnTest)+'.pkl'
    pickle.dump(res, open(resFile, 'wb'))

