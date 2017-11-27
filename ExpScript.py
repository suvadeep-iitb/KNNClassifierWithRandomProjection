#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
import labelCount as lc
#from KNNPredictor import KNNPredictor as KNNPredictor
from RandomEmbeddingAKNNPredictor import RandomEmbeddingAKNNPredictor as KNNPredictor


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
  "numLearners": 1, # Currently works for only 1
  "numThreads": 1,
  "embDim": 20,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "lamb": 1,
  "maxTestSamples": 4000,
  "maxTrainSamples": 100000}

lambdaList = [0.1]
nnTestList = [3, 5, 10]
embDimList = [100]
maxTS = [200000000]

for i in [1]:
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

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]

  # Normalize data
  data.X, data.Xt = MyNormalize(data.X, data.Xt, params['normalization'])

  resFilePrefix = labelStruct.resFile;
  for ts in maxTS:
    params['maxTrainSamples'] = ts
    for lam in lambdaList:
      for ed in embDimList:
        params["lamb"] = lam
        params["embDim"] = ed
        print("\tRunning for " + "lambda = " + str(params["lamb"]) + " emb_dim = " + str(params["embDim"]));

        knnPredictor = KNNPredictor(params)
        knnPredictor.Train(data.X, 
                         data.Y, 
                         params['maxTrainSamples'], 
                         params['numThreads'])
        testResList = knnPredictor.PredictAndComputePrecision(
                         data.Xt,
                         data.Yt,
                         nnTestList,
                         params['maxTestSamples'],
                         max(params['numThreads'], 25))
        trainResList = knnPredictor.PredictAndComputePrecision(
                         data.X,
                         data.Y,
                         nnTestList,
                         params['maxTestSamples'],
                         max(params['numThreads'], 25))
        resFile = 'Results/RandProj_'+resFilePrefix+'_TS'+str(ts)+'_L'+str(lam)+'_D'+str(ed)+'.pkl'
        pickle.dump({'testRes' : testResList, 
                     'trainRes' : trainResList, 
                     'nnTestList' : nnTestList, 
                     'featureProjMatrix' : knnPredictor.featureProjMatrix,
                     'trainSample' : knnPredictor.sampleIndices,
                     'trainError' : knnPredictor.trainError,
                     'testError' : knnPredictor.MeanSquaredError(data.Xt, data.Yt, params['maxTestSamples']),
                     'params' : params}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('')
        print('')
        print('')
