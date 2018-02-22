import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
#from sklearn.cluster import KMeans as kmeans
#from sklearn.cluster import MiniBatchKMeans as kmeans
from MyCluster import NearestNeighbour
import labelCount as lc
from MulticlassPredictor import MulticlassPredictor as MulticlassPredictor
#from RandomEmbeddingMulticlassPredictor import RandomEmbeddingMulticlassPredictor as MulticlassPredictor
from ClusteredMulticlassPredictor import ClusteredMulticlassPredictor


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

  multiclassPredictor = ClusteredMulticlassPredictor(p)
  newParam = p.copy()
  multiclassPredictor.Train(data.X, 
                     data.Y, 
                     maxTrainSamples = p['maxTrainSamples'], 
                     numThreads = nt)
  testResList = multiclassPredictor.PredictAndComputePrecision(
                     data.Xt,
                     data.Yt,
                     maxTestSamples = p['maxTestSamples'],
                     numThreads = 30)
  '''
  trainResList = multiclassPredictor.PredictAndComputePrecision(
                     data.X,
                     data.Y,
                     maxTestSamples = p['maxTestSamples'],
                     numThreads = 30)
  '''
  resFile = 'Results/ClusteredMulticlass_'+p['resFilePrefix']+'_TS'+str(ts)+'_CL'+str(nc)+'_L'+str(lamb)+'_D'+str(ed)+'.pkl'
  del p['clusteringAlgo']
  pickle.dump({'testRes' : testResList, 
               #'trainRes' : trainResList, 
               #'featureProjMatrix' : multiclassPredictor.GetFeatureProjMatrix(),
               #'labelProjMatrix' : multiclassPredictor.GetLabelProjMatrix(),
               #'trainSample' : multiclassPredictor.sampleIndices,
               'params' : p}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
  print('Finished')
  print('')
  print('')


params = {
  "numLearners": 1,
  "numThreads": 20,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "seed": 1,
  "itr": 10,
  "logFile": '',
  "maxTestSamples": 500000}

embDimList = [100, 500]
clusterSizeList = [50000]
lambdaList = [0.0001, 0.001, 0.01, 0.1]


maxTS = [0]

for i in [6]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print("Running for " + dataFile)
  data = pickle.load(open(dataFile, 'rb'))
  # For related search data, feature matrix in dense

  # Perform initial random permutation of the data
  '''
  print("Randomly permuting the data ...")
  perm = np.random.permutation(data.X.shape[0])
  data.X = data.X[perm, :]
  data.Y = data.Y[perm, :]

  perm = np.random.permutation(data.Xt.shape[0])
  data.Xt = data.Xt[perm, :]
  data.Yt = data.Yt[perm, :]
  '''

  # Remove label with no sample
  labelCounts = np.array(np.sum(data.Y, axis=0)).reshape(-1)
  nonemptyLabels = (labelCounts > 0)
  data.Y = data.Y[:, nonemptyLabels]
  data.Yt = data.Yt[:, nonemptyLabels]

  # Normalize data
  data.X, data.Xt = MyNormalize(data.X, data.Xt, params['normalization'])

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]
  params["resFilePrefix"] = labelStruct.resFile

  paramList = []
  for ed in embDimList:
    for ts in maxTS:
      for clusSize in clusterSizeList:
        for lamb in lambdaList:
          newParams = params.copy()
          newParams['maxTrainSamples'] = ts
          newParams['lamb'] = lamb
          newParams['numClusters'] = int(data.X.shape[0]/clusSize)
          newParams['embDim'] = ed
          newParams['clusteringAlgo'] = NearestNeighbour(n_clusters = newParams['numClusters'],
                                                         max_iter = 10,
                                                         num_nn = 10,
                                                         label_normalize = 1,
                                                         eta0 = 0.1,
                                                         lamb = 4.0,
                                                         seed = newParams['seed'],
                                                         verbose = 1)
          newParams['basePredictor'] = MulticlassPredictor(newParams)
          paramList.append(newParams)

  for p in paramList:
    PerformExperiment(p, data) 
