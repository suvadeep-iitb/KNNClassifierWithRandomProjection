#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.preprocessing import normalize
import labelCount as lc
from MulticlassPredictor import MulticlassPredictor


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
  "numThreads": 40,
  "embDim": 15,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "lamb": 1,
  "seed": 1,
  "maxTestSamples": 10000,
  "maxTrainSamples": 600000}

lambdaList = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
#lambdaList = [1]
maxTS = [0]

for i in [15, 2]:
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

  resFilePrefix = labelStruct.resFile;
  for ts in maxTS:
    params['maxTrainSamples'] = ts
    for lam in lambdaList:
        params["lamb"] = lam
        print("\tRunning for " + "lambda = " + str(params["lamb"]) + " emb_dim = " + str(params["embDim"]));

        multiclassPredictor = MulticlassPredictor(params)
        multiclassPredictor.Train(data.X, 
                         data.Y,
                         maxTrainSamples = params['maxTrainSamples'],
                         numThreads = params['numThreads'])
        testRes = multiclassPredictor.PredictAndComputePrecision(
                         data.Xt,
                         data.Yt,
                         maxTestSamples = params['maxTestSamples'],
                         numThreads = max(params['numThreads'], 15))
        '''
        trainRes = multiclassPredictor.PredictAndComputePrecision(
                         data.X,
                         data.Y,
                         maxTestSamples = params['maxTestSamples'],
                         numThreads = max(params['numThreads'], 15))
        '''
        resFile = 'Results/Multiclass_'+resFilePrefix+'_TS'+str(ts)+'_L'+str(lam)+'.pkl'
        pickle.dump({'testRes' : testRes,
                     #'trainRes' : trainRes,
                     #'paramMatrix' : multiclassPredictor.W,
                     #'trainSample' : multiclassPredictor.sampleIndices,
                     #'trainError' : multiclassPredictor.trainError,
                     #'testError' : multiclassPredictor.MeanSquaredError(data.Xt, data.Yt, params['maxTestSamples']),
                     'params' : params}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('')
        print('')
        print('')
