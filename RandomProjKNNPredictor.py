import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, hstack, vstack, issparse
from collections import namedtuple
import pickle
from joblib import Parallel, delayed
import multiprocessing
import nmslib
import math
from liblinearutil import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVR
import threading
from datetime import datetime

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

  numCores = numThreads; # multiprocessing.cpu_count()
  resultList = Parallel(n_jobs = numCores)(delayed(TrainWrapper)(Z[:, l], X, l, C) for l in range(embDim))

  # Collect the model parameters into a matrix
  for l in range(embDim):    
    W[:, l] = resultList[l]
  return W
  


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



def ComputeAKNN(index, W, Xt, nnTest, numThreads):
  # Project the Xt into the embedding space
  if(issparse(Xt)):
    pXt = Xt * W
  else:
    pXt = np.matmul(Xt, W);

  # get the nearest neighbours for all the test datapoint
  neighbors = index.knnQueryBatch(pXt, nnTest, num_threads=numThreads)
  
  # Create the KNN matrix
  AKNN = np.zeros((pXt.shape[0], nnTest), dtype=np.int64);
  for i,nei in enumerate(neighbors):
    if (len(nei[0]) < nnTest):
      print(str(pXt.shape[0])+'/'+str(i)+' '+str(len(nei[0]))+' '+str(len(nei[1])))
    AKNN[i, :] = nei[0]

  return AKNN



def ComputeAKNN(W, X, Xt, nnTest):
  nt = Xt.shape[0]

  # Project the X and Xt into the embedding space
  if(issparse(X)):
    pX = X * W;
    pXt = Xt * W;
  else:
    pX = np.matmul(X, W)
    pXt = np.matmul(Xt, W);

  batchSize = 1000
  numBatch = int(math.ceil(float(nt)/batchSize))
  KNN = np.zeros((nt, nnTest), dtype=np.uint64)
  
  for b in range(numBatch):
    sIdx = b*batchSize
    eIdx = min((b+1)*batchSize, nt)
    dist = euclidean_distances(pXt[sIdx: eIdx], pX)
    neighbors = np.argpartition(dist, nnTest)[:, :nnTest]
    dist_temp = np.zeros(neighbors.shape)
    for i in range(eIdx - sIdx):
      for j in range(nnTest):
        dist_temp[i, j] = dist[i, neighbors[i, j]]
    dist = dist_temp
    sortedIdx = np.argsort(dist)
    for i in range(eIdx - sIdx):
      for j in range(nnTest):
        KNN[sIdx + i, j] = neighbors[i, sortedIdx[i, j]]

  return KNN



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
    elif (t[0] == rowIdx+1):
      rowIdx += 1
      sortedIdx[rowIdx, 0] = t[1]
      sortedVal[rowIdx, 0] = -t[2]
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
      if (Yt[i, predYt[i, j]] > 0):
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



def SaveModel(W, filename):
  D, d = W.shape
  with open(filename, 'w') as fout:
    fout.write(str(D) + ' ' + str(d) + '\n')
    for r in range(D):
      for c in range(d):
        fout.write(str(W[r, c]) + ' ');
      fout.write('\n');




def LoadModel(filename):
  with open(filename) as fin:
    line = fin.readline().split();
    D = int(line[0])
    d = int(line[1])
    W = np.zeros((D, d))
    for r in range(D):
      line = fin.readline().split()
      assert(len(line) == d)
      for c in range(d):
        W[r, c] = float(line[c]);
  return W



def RandomProjAKNNPredictor(X, Y, Xt, Yt, params, nnTestList):
  # Make sure label index is stating from 1
  if ((Y[:, 0].nnz != 0) or (Yt[:, 0].nnz != 0)):
    print(str(datetime.now()) + " : " + "Pre-pending zero column in the label matrices")
    Y = csr_matrix(hstack((csr_matrix((Y.shape[0], 1)), Y)))
    Yt = csr_matrix(hstack((csr_matrix((Yt.shape[0], 1)), Yt)))

  maxTestSamples = params["maxTestSamples"]
  maxTrainSamples= params["maxTrainSamples"]
  print(str(datetime.now()) + " : " + "Performing down-sampling")
  # Sample test data for faster compuation of test precision
  Xt, Yt = DownSampleData(Xt, Yt, maxTestSamples)
  # Sample train data for faster training
  X_sam, Y_sam = DownSampleData(X, Y, maxTrainSamples)
  # sample train data for faster computation of train precision
  X_sam_t, Y_sam_t = DownSampleData(X, Y, maxTestSamples)

  print(str(datetime.now()) + " : " + "Starting training")
  W = RandProj(X_sam, Y_sam, params);

  print(str(datetime.now()) + " : " + "Training Finished")

  maxNNTest = max(nnTestList);
  numCores = multiprocessing.cpu_count()
  numThreads = max(params["numThreads"], int(2*numCores/3));

  # Create K nearest neighbor graph over training examples
  print(str(datetime.now()) + " : " + "Creating Approximate KNN graph over train examples")
  graph = CreateAKNNGraph(W, X, numThreads)

  # Compute K nearest neighbors for sampled test examples
  print(str(datetime.now()) + " : " + "Computing Approximate KNN of test examples")
  KNN = ComputeAKNN(graph, W, Xt, maxNNTest, numThreads);

  # Compute K nearest neighbors for sampled train examples
  print(str(datetime.now()) + " : " + "Computing Approximate KNN of training examples")
  KNN_tr = ComputeAKNN(graph, W, X_sam_t, maxNNTest, numThreads);

  for nnt in range(len(nnTestList)):
    nnTest = nnTestList[nnt]

    # Predict labels for sampled test data
    print(str(datetime.now()) + " : " + "Performing prediction on sampled test set")
    predYt, scoreYt = PredictYParallel(Y, KNN, nnTest, numThreads)

    # Compute precisions for sampled test data
    print(str(datetime.now()) + " : " + "Computing precisions for sampled test set")
    precision = ComputePrecisionParallel(predYt, Yt, 5, numThreads)

    # Predict labels for sampled train data
    print(str(datetime.now()) + " : " + "Performing prediction on sampled train set")
    predYt_tr, scoreYt_tr = PredictYParallel(Y, KNN_tr, nnTest, numThreads)

    # Compute precisions for sampled train data
    print(str(datetime.now()) + " : " + "Computing precisions for sampled train set")
    precision_tr = ComputePrecisionParallel(predYt_tr, Y_sam_t, 5, numThreads)

    # Save result
    print(str(datetime.now()) + " : " + "Saving Results")
    res = {}
    res["precision"] = precision
    #res["predictedLabel"] = predYt
    #res["score"] = scoreYt
    res["precision_tr"] = precision_tr
    #res["predictedLabel_tr"] = predYt_tr
    #res["score_tr"] = scoreYt_tr

    resFile = 'Results/'+params["resFilePrefix"]+'_L'+str(params["lamb"])+'_D'+str(params["embDim"])+'_NN'+str(nnTest)+'.pkl'
    pickle.dump(res, open(resFile, 'wb'))

