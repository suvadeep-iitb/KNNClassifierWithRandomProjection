#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, issparse
from sklearn.preprocessing import normalize
import labelCount as lc
#from KNNPredictor import KNNPredictor as KNNPredictor
from RandomEmbeddingAKNNPredictor import RandomEmbeddingAKNNPredictor as KNNPredictor
from EnsembleKNNPredictor import EnsembleKNNPredictor
#from OneVsRestEmbeddingAKNNPredictor import OneVsRestEmbeddingAKNNPredictor as KNNPredictor


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

params = {
  "numLearners": 5,
  "numThreads": 5,
  "embDim": 15,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "lamb": 1,
  "seed": 1,
  "logFile": '',
  "maxTestSamples": 1000000,
  "maxTrainSamples": 6000000}


lambdaList = [0.0001, 0.001, 0.01]
nnTestList = [10, 20]
embDimList = [20, 50]
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

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]


  # Normalize data
  data.X, data.Xt = MyNormalize(data.X, data.Xt, params['normalization'])

  # Append a constant column to the feature matrix
  #data.X = hstack((data.X, csr_matrix(np.ones((data.X.shape[0], 1), dtype=float))/np.sqrt(data.X.shape[1])))
  #data.Xt = hstack((data.Xt, csr_matrix(np.ones((data.Xt.shape[0], 1), dtype=float))/np.sqrt(data.Xt.shape[1])))

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]

  resFilePrefix = labelStruct.resFile;
  for ed in embDimList:
    for ts in maxTS:
      params['maxTrainSamples'] = ts
      for lam in lambdaList:
        params["lamb"] = lam
        params["embDim"] = ed
        print("\tRunning for " + "lambda = " + str(params["lamb"]) + " emb_dim = " + str(params["embDim"]));

        params["basePredictor"] = KNNPredictor(params)
        knnPredictor = EnsembleKNNPredictor(params)
        knnPredictor.Train(data.X, 
                         data.Y,
                         numThreads = params['numThreads'])
        testResList = knnPredictor.PredictAndComputePrecision(
                         data.Xt,
                         data.Yt,
                         nnTestList,
                         params['maxTestSamples'],
                         max(params['numThreads'], 30))
        '''
        trainResList = knnPredictor.PredictAndComputePrecision(
                         data.X,
                         data.Y,
                         nnTestList,
                         params['maxTestSamples'],
                         max(params['numThreads'], 30))
        '''
        #resFile = 'Results/OvRRP_'+resFilePrefix+'_TS'+str(ts)+'_L'+str(lam)+'_D'+str(ed)+'.pkl'
        #resFile = 'Results/KNN_'+resFilePrefix+'_TS'+str(ts)+'_L'+str(lam)+'_D'+str(ed)+'.pkl'
        resFile = 'Results/RandProj_'+resFilePrefix+'_TS'+str(ts)+'_L'+str(lam)+'_D'+str(ed)+'.pkl'
        pickle.dump({'testRes' : testResList, 
                     #'trainRes' : trainResList, 
                     'nnTestList' : nnTestList, 
                     #'featureProjMatrix' : knnPredictor.GetFeatureProjMatrix(),
                     #'trainSample' : knnPredictor.sampleIndices,
                     #'trainError' : knnPredictor.trainError,
                     #'testError' : knnPredictor.MeanSquaredError(data.Xt, data.Yt, params['maxTestSamples']),
                     'params' : params}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('')
        print('')
        print('')
