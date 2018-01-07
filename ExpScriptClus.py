#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
#from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import MiniBatchKMeans as kmeans
import labelCount as lc
#from MultipleOrthogonalBinaryClusteringAKNNPredictor import MultipleOrthogonalBinaryClusteringAKNNPredictor as KNNPredictor
from RandomEmbeddingAKNNPredictor import RandomEmbeddingAKNNPredictor as KNNPredictor
from ClusteredKNNPredictor import ClusteredKNNPredictor
from EnsembleKNNPredictor import EnsembleKNNPredictor
from joblib import Parallel, delayed
import multiprocessing


#Data = namedtuple("Data", "X Y Xt Yt")
class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt

def MyNormalize(X, Xt, norm):
  print("Normalizing data ...")
  n = X.shape[0]
  if issparse(X):
    XXt = vstack([X, Xt]) 
  else:
    XXt = np.vstack([X, Xt])
  assert(norm in ['l2_row', 'l2_col', 'l1_row', 'l1_col', 'max_row', 'max_col'])
  if 'row' in norm:
    axis = 1
  else:
    axis = 0
  if 'l2' in norm:
    nor = 'l2'
  elif 'l1' in norm:
    nor = 'l1'
  else:
    nor = 'max'
  XXt = normalize(XXt, norm = nor, axis = axis)
  print("Normalization done")
  return XXt[:n, :], XXt[n:, :]


def PerformExperiment(p, data):
  '''
  mu1 = p['mu1']
  mu2 = p['mu2']
  mu3 = p['mu3']
  mu4 = p['mu4']
  it = p['outerIter']
  '''
  lamb = p['lamb']
  ed = p['embDim']
  nc = p['numClusters']
  #print("Running for " + "mu1 = " + str(mu1)  + " mu2 = " + str(mu2) + " mu3 = " + str(mu3) + " mu4 = " + str(mu4) + " emb_dim = " + str(ed) + "  iter = " + str(it));
  print("Running for train_sam = " + str(ts) + " lambda = " + str(lamb)  + " emb_dim = " + str(ed) + " # clusters = " + str(nc));

  knnPredictor = ClusteredKNNPredictor(p)
  newParam = p.copy()
  newParam['basePredictor'] = knnPredictor
  ensembleKNNPredictor = knnPredictor # EnsembleKNNPredictor(newParam)
  ensembleKNNPredictor.Train(data.X, 
                     data.Y, 
                     maxTrainSamples = p['maxTrainSamples'], 
                     numThreads = 1)
  testResList = ensembleKNNPredictor.PredictAndComputePrecision(
                     data.Xt,
                     data.Yt,
                     p["nnTestList"],
                     p['maxTestSamples'],
                     numThreads = 1)
  '''
  trainResList = ensembleKNNPredictor.PredictAndComputePrecision(
                     data.X,
                     data.Y,
                     p["nnTestList"],
                     p['maxTestSamples'],
                     numThreads = 1)
  '''
  #resFile = 'Results/ClusteredMOBCAP_'+p['resFilePrefix']+'_TS'+str(ts)+'_CL'+str(nc)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_MU4'+str(mu4)+'_D'+str(ed)+'_IT'+str(it)+'.pkl'
  resFile = 'Results/ClusteredRandProj_'+p['resFilePrefix']+'_TS'+str(ts)+'_CL'+str(nc)+'_L'+str(lamb)+'_D'+str(ed)+'.pkl'
  pickle.dump({'testRes' : testResList, 
               #'trainRes' : trainResList, 
               'nnTestList' : p['nnTestList'], 
               #'featureProjMatrix' : knnPredictor.GetFeatureProjMatrix(),
               #'labelProjMatrix' : knnPredictor.GetLabelProjMatrix(),
               'trainSample' : knnPredictor.sampleIndices,
               'params' : p}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
  print('Finished')
  print('')
  print('')


params = {
  "numLearners": 1,
  "numClusters": 5,
  "numThreads": 2,
  #"embDim": 20,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  #"mu1": 1,
  #"mu2": 1,
  #"mu3": 1,
  #"mu4": 1,
  #"innerIter": 8,
  #"outerIter": 3,
  "seed": 1,
  "maxTestSamples": 50000,
  #"maxTrainSamples": 600000,
  "clusteringAlgo": kmeans,
  "basePredictor": KNNPredictor}
'''
outerIterList = [3]
mu1List = [1]
mu2List = [1]
mu3List = [1]
mu4List = [0]
'''

nnTestList = [10, 20]
embDimList = [20]
numClustersList = [10, 50, 100, 500]
lambdaList = [0.0001, 0.001, 0.01, 0.1, 1]

maxTS = [0]

for i in [4]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print("Running for " + dataFile)
  data = pickle.load(open(dataFile, 'rb'))
  # For related search data, feature matrix in dense

  # Perform initial random permutation of the data
  print("Randomly permuting the data ...")
  perm = np.random.permutation(data.X.shape[0])
  data.X = data.X[perm, :]
  data.Y = data.Y[perm, :]

  perm = np.random.permutation(data.Xt.shape[0])
  data.Xt = data.Xt[perm, :]
  data.Yt = data.Yt[perm, :]

  # Remove label with no sample
  labelCounts = np.array(np.sum(data.Y, axis=0)).reshape(-1)
  nonemptyLabels = (labelCounts > 0)
  data.Y = data.Y[:, nonemptyLabels]
  data.Yt = data.Yt[:, nonemptyLabels]

  # Normalize data
  data.X, data.Xt = MyNormalize(data.X, data.Xt, params['normalization'])

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]
  params["nnTestList"] = nnTestList
  params["resFilePrefix"] = labelStruct.resFile

  paramList = []
  for ed in embDimList:
    for ts in maxTS:
      for numClusters in numClustersList:
        for lamb in lambdaList:
          newParams = params.copy()
          newParams['maxTrainSamples'] = ts
          newParams['lamb'] = lamb
          newParams['numClusters'] = numClusters
          newParams["embDim"] = ed
          newParams["logFile"] = ''#'Results/MOBCAP_'+params['resFilePrefix']+'_log_TS'+str(ts)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_MU4'+str(mu4)+'_D'+str(ed)+'_IT'+str(it)
          paramList.append(newParams)

  numThreads = 15 #params['numThreads']
  Parallel(n_jobs = numThreads)(delayed(PerformExperiment)(p, data) for p in paramList)
