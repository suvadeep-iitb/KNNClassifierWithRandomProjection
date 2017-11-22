#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
import labelCount as lc
from MultipleOrthogonalBinaryClusteringAKNNPredictor import MultipleOrthogonalBinaryClusteringAKNNPredictor as KNNPredictor


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
  "embDim": 100,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "mu12": 1,
  "mu3": 1,
  "innerIter": 1,
  "outerIter": 10,
  "seed": 1,
  "maxTestSamples": 2000000,
  "maxTrainSamples": 10000000}

mu12List = [1]
mu3List = [1]
nnTestList = [3, 5, 10]
embDimList = [20]
maxTS = [0]

for i in [1]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print("Running for " + dataFile)
  data = pickle.load(open(dataFile, 'rb'))
  # For related search data, feature matrix in dense

  # Perform initial random permutation of the data
  print("Randomly permuting the data ...")
  perm = np.random.permutation(data.X.shape[0])
  data.X = data.X[perm, :][:100, :100]
  data.Y = data.Y[perm, :][:100, :100]

  perm = np.random.permutation(data.Xt.shape[0])
  data.Xt = data.Xt[perm, :][:100, :100]
  data.Yt = data.Yt[perm, :][:100, :100]

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]

  # Normalize data
  data.X, data.Xt = MyNormalize(data.X, data.Xt, params['normalization'])

  resFilePrefix = labelStruct.resFile;
  for ts in maxTS:
    params['maxTrainSamples'] = ts
    for mu12 in mu12List:
      for mu3 in mu3List:
        for ed in embDimList:
          params["mu1"] = mu12
          params["mu2"] = mu12
          params["mu3"] = mu3
          params["embDim"] = ed
          params["paramSaveFile"] = 'Results/'+resFilePrefix+'_log__TS'+str(ts)+'_MU1'+str(mu12)+'_MU2'+str(mu12)+'_MU3'+str(mu3)+'_D'+str(ed)+'.pkl'
          print("Running for " + "mu1 = " + str(params["mu1"])  + "mu2 = " + str(params["mu2"]) + "mu3 = " + str(params["mu3"]) + " emb_dim = " + str(params["embDim"]));

          knnPredictor = KNNPredictor(params)
          knnPredictor.Train(data.X, 
                             data.Y, 
                             params['outerIter'],
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
          resFile = 'Results/'+resFilePrefix+'_TS'+str(ts)+'_MU1'+str(mu12)+'_MU2'+str(mu12)+'_MU3'+str(mu3)+'_D'+str(ed)+'.pkl'
          pickle.dump({'testRes' : testResList, 
                       'trainRes' : trainResList, 
                       'nnTestList' : nnTestList, 
                       'featureProjMatrix' : knnPredictor.featureProjMatrix,
                       'labelProjMatrix' : knnPredictor.labelProjMatrix,
                       'trainSample' : knnPredictor.sampleIndices,
                       'params' : params}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
          print('')
          print('')
          print('')
