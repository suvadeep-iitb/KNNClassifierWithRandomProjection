#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
import labelCount as lc
from MultipleOrthogonalBinaryClusteringAKNNPredictor import MultipleOrthogonalBinaryClusteringAKNNPredictor as KNNPredictor
from EnsembleKNNPredictor import EnsembleKNNPredictor


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
  "numLearners": 1,
  "numThreads": 15,
  "embDim": 20,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "mu1": 1,
  "mu2": 1,
  "mu3": 1,
  "innerIter": 1,
  "outerIter": 8,
  "seed": 1,
  "maxTestSamples": 2000000,
  "maxTrainSamples": 10000000,
  "basePredictor": KNNPredictor}

'''
mu1List = [0.01, 1, 100, 10000, 1000000]
mu2List = [0.000001, 0.0001, 0.01, 1, 100]
mu3List = [0.0001, 0.01, 1, 100, 10000]
'''
mu1List = [1]
mu2List = [0.01]
mu3List = [1]
nnTestList = [10, 15, 20]
embDimList = [20]
maxTS = [0]

for i in [2]:
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
  for ed in embDimList:
    for ts in maxTS:
      params['maxTrainSamples'] = ts
      for mu1 in mu1List:
        for mu2 in mu2List:
          for mu3 in mu3List:
            params["mu1"] = mu1
            params["mu2"] = mu2
            params["mu3"] = mu3
            params["embDim"] = ed
            params["logFile"] = '../Results/MOBCAP_'+resFilePrefix+'_log_TS'+str(ts)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_D'+str(ed)
            print("Running for " + "mu1 = " + str(params["mu1"])  + " mu2 = " + str(params["mu2"]) + " mu3 = " + str(params["mu3"]) + " emb_dim = " + str(params["embDim"]));

            knnPredictor = EnsembleKNNPredictor(params)
            knnPredictor.Train(data.X, 
                               data.Y, 
                               maxTrainSamples = params['maxTrainSamples'], 
                               numThreads = params['numThreads'],
                               itr = params['outerIter'])
            testResList = knnPredictor.PredictAndComputePrecision(
                               data.Xt,
                               data.Yt,
                               nnTestList,
                               params['maxTestSamples'],
                               max(params['numThreads'], 15))
            trainResList = knnPredictor.PredictAndComputePrecision(
                               data.X,
                               data.Y,
                               nnTestList,
                               params['maxTestSamples'],
                               max(params['numThreads'], 15))
            resFile = '../Results/MOBCAP_'+resFilePrefix+'_TS'+str(ts)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_D'+str(ed)+'.pkl'
            pickle.dump({'testRes' : testResList, 
                         'trainRes' : trainResList, 
                         'nnTestList' : nnTestList, 
                         'featureProjMatrix' : knnPredictor.GetFeatureProjMatrix(),
                         'labelProjMatrix' : knnPredictor.GetLabelProjMatrix(),
                         'trainSample' : knnPredictor.sampleIndices,
                         'params' : params}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
          print('')
          print('')
          print('')
