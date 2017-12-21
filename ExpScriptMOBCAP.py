#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
import labelCount as lc
from MultipleOrthogonalBinaryClusteringAKNNPredictor import MultipleOrthogonalBinaryClusteringAKNNPredictor as KNNPredictor
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
  mu1 = p['mu1']
  mu2 = p['mu2']
  mu3 = p['mu3']
  mu4 = p['mu4']
  ed = p['embDim']
  it = p['outerIter']
  print("Running for " + "mu1 = " + str(mu1)  + " mu2 = " + str(mu2) + " mu3 = " + str(mu3) + " mu4 = " + str(mu4) + " emb_dim = " + str(ed) + "  iter = " + str(it));

  knnPredictor = EnsembleKNNPredictor(p)
  knnPredictor.Train(data.X, 
                     data.Y, 
                     maxTrainSamples = p['maxTrainSamples'], 
                     numThreads = p['numThreads'],
                     itr = p['outerIter'])
  testResList = knnPredictor.PredictAndComputePrecision(
                     data.Xt,
                     data.Yt,
                     p["nnTestList"],
                     p['maxTestSamples'],
                     max(p['numThreads'], 10))
  '''
  trainResList = knnPredictor.PredictAndComputePrecision(
                     data.X,
                     data.Y,
                     p["nnTestList"],
                     p['maxTestSamples'],
                     max(p['numThreads'], 10))
  '''
  resFile = 'Results/MOBCAP_'+p['resFilePrefix']+'_TS'+str(ts)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_MU4'+str(mu4)+'_D'+str(ed)+'_IT'+str(it)+'.pkl'
  pickle.dump({'testRes' : testResList, 
               #'trainRes' : trainResList, 
               'nnTestList' : p['nnTestList'], 
               'featureProjMatrix' : knnPredictor.GetFeatureProjMatrix(),
               'labelProjMatrix' : knnPredictor.GetLabelProjMatrix(),
               'trainSample' : knnPredictor.sampleIndices,
               'params' : p}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
  print('Finished')
  print('')
  print('')


params = {
  "numLearners": 1,
  "numThreads": 30,
  #"embDim": 20,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  #"mu1": 1,
  #"mu2": 1,
  #"mu3": 1,
  #"mu4": 1,
  "innerIter": 2,
  "outerIter": 3,
  "seed": 1,
  "maxTestSamples": 50000,
  #"maxTrainSamples": 600000,
  "basePredictor": KNNPredictor}
outerIterList = [5, 7, 10]
mu1List = [0.000001, 0.0001, 0.01, 1]
mu2List = [1, 100, 10000, 1000000]
mu3List = [0.0001, 0.01, 1, 100, 10000]
'''
mu1List = [1]
mu2List = [1]
mu3List = [1]
'''

mu4List = [0]
nnTestList = [20]
embDimList = [50]
maxTS = [0]

for i in [14]:
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
  params["initialFeatureProjMatrixFile"] = labelStruct.initialFeatureProjMatrixFile
  params["resFilePrefix"] = labelStruct.resFile

  paramList = []
  for ed in embDimList:
    for ts in maxTS:
      for mu1 in mu1List:
        for mu2 in mu2List:
          for mu3 in mu3List:
            for mu4 in mu4List:
              for it in outerIterList:
                newParams = params.copy()
                newParams['maxTrainSamples'] = ts
                newParams["mu1"] = mu1
                newParams["mu2"] = mu2
                newParams["mu3"] = mu3
                newParams["mu4"] = mu4
                newParams["embDim"] = ed
                newParams["outerIter"] = it
                newParams["logFile"] = ''#'Results/MOBCAP_'+params['resFilePrefix']+'_log_TS'+str(ts)+'_MU1'+str(mu1)+'_MU2'+str(mu2)+'_MU3'+str(mu3)+'_MU4'+str(mu4)+'_D'+str(ed)+'_IT'+str(it)
                paramList.append(newParams)

  numThreads = params['numThreads']
  Parallel(n_jobs = numThreads)(delayed(PerformExperiment)(p, data) for p in paramList)

