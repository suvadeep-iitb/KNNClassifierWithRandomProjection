#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
#from sklearn.cluster import KMeans as kmeans
from sklearn.cluster import MiniBatchKMeans as kmeans
from MyCluster import NearestNeighbour, LabelNeighbourExpansionEP
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
  lamb = p['lamb']
  ed = p['embDim']
  nc = p['numClusters']
  nt = p['numThreads']
  print("Running for train_sam = " + str(ts) + " lambda = " + str(lamb)  + " emb_dim = " + str(ed) + " # of clusters = " + str(nc));

  knnPredictor = ClusteredKNNPredictor(p)
  newParam = p.copy()
  newParam['basePredictor'] = knnPredictor
  ensembleKNNPredictor = knnPredictor # EnsembleKNNPredictor(newParam)
  knnPredictor.Train(data.X, 
                     data.Y, 
                     maxTrainSamples = p['maxTrainSamples'], 
                     numThreads = nt)
  testResList = knnPredictor.PredictAndComputePrecision(
                     data.Xt,
                     data.Yt,
                     p["nnTestList"],
                     p['maxTestSamples'],
                     numThreads = 20)
  '''
  trainResList = knnPredictor.PredictAndComputePrecision(
                     data.X,
                     data.Y,
                     p["nnTestList"],
                     p['maxTestSamples'],
                     numThreads = 20)
  '''
  resFile = 'Results/ClusteredRandProj_'+p['resFilePrefix']+'_TS'+str(ts)+'_CL'+str(nc)+'_L'+str(lamb)+'_D'+str(ed)+'.pkl'
  del p['clusteringAlgo']
  pickle.dump({'testRes' : testResList, 
               #'trainRes' : trainResList, 
               'nnTestList' : p['nnTestList'], 
               #'featureProjMatrix' : knnPredictor.GetFeatureProjMatrix(),
               #'labelProjMatrix' : knnPredictor.GetLabelProjMatrix(),
               #'trainSample' : knnPredictor.sampleIndices,
               'params' : p}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
  print('Finished')
  print('')
  print('')


params = {
  "numLearners": 1,
  "numThreads": 20,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "seed": 1,
  "logFile": '',
  "maxTestSamples": 5000000,
  "basePredictor": KNNPredictor}

nnTestList = [10]
embDimList = [50]
numClustersList = [300]
lambdaList = [0.01, 0.1, 1]

maxTS = [0]

for i in [6]:
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
          newParams['embDim'] = ed
          center_file = 'ClusAss/res_wikiLSHTC_centers_C300_S0_NN10_C1.pkl'
          newParams['clusteringAlgo'] = LabelNeighbourExpansionEP(
                                                         n_clusters = numClusters,
                                                         seed = newParams['seed'],
                                                         n_jobs = 20,
                                                         verbose = 1)
          newParams['basePredictor'] = KNNPredictor(newParams)
          newParams["logFile"] = ''#'Results/MOBCAP_'+params['resFilePrefix']+'_log_TS'+str(ts)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_MU4'+str(mu4)+'_D'+str(ed)+'_IT'+str(it)
          paramList.append(newParams)

  for p in paramList:
    PerformExperiment(p, data) 
