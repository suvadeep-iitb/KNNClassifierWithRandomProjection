import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, issparse
from sklearn.preprocessing import normalize
from RandomEmbeddingAKNNPredictor import RandomEmbeddingAKNNPredictor as KNNPredictor
from EnsembleKNNPredictor import EnsembleKNNPredictor
import sys


# Set the hyper parameters here
###############################
numThreads = 10             # number of core to be used in regression. WARNING: increasing
                            # this value also increases the memory usage as the
                            # entire train dataset is copied for each processe

numLearners = 5             # number of predictor in the ensemble learner

maxTestSamples = 0          # maximum size test sample to be used during the computation
                            # of precisions. Put 0 or any value larger than the size of
                            # testset to use the entire testset during prediction

seed = 0                    # seed to be used for random sampling 
 
lambdaList = [0.01, 0.1]    # list of lambda values for which the experiments will be
                            # repeated. NOTE: lambda is the hyperpermeter used to set
                            # relative weightage of panalty term and regularization
                            # term in the loss of liblinear 

embDimList = [20, 50]       # list of embidding dimensions for which the experiments
                            # to be repeated

nnTestList = [5, 10]        # list of the numbers of the nearest neighbours to be used
                            # during the prediction
###############################


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
  "numLearners": numLearners,
  "numThreads": numThreads,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "seed": seed,
  "logFile": '',
  "maxTestSamples": maxTestSamples,
  "maxTrainSamples": 0}

maxTS = [0]

if __name__ == '__main__':
  dataFile = sys.argv[1]
  resFilePrefix = sys.argv[2]

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

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]

  for ed in embDimList:
    for ts in maxTS:
      params['maxTrainSamples'] = ts
      for lam in lambdaList:
        params["lamb"] = lam
        params["embDim"] = ed
        print("\nRunning for " + "lambda = " + str(params["lamb"]) + " emb_dim = " + str(params["embDim"]));

        params['basePredictor'] = KNNPredictor(params)
        knnPredictor = EnsembleKNNPredictor(params)
        knnPredictor.Train(data.X, 
                         data.Y,
                         maxTrainSamples = params['maxTrainSamples'],
                         numThreads = params['numThreads'])
        testResList = knnPredictor.PredictAndComputePrecision(
                         data.Xt,
                         data.Yt,
                         nnTestList,
                         params['maxTestSamples'],
                         max(params['numThreads'], 30))
        trainResList = knnPredictor.PredictAndComputePrecision(
                         data.X,
                         data.Y,
                         nnTestList,
                         params['maxTestSamples'],
                         max(params['numThreads'], 40))
        resFile = resFilePrefix+'_L'+str(lam)+'_D'+str(ed)+'.pkl'
        pickle.dump({'testRes' : testResList, 
                     'trainRes' : trainResList, 
                     'nnTestList' : nnTestList, 
                     'params' : params}, open(resFile, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('')
        print('')
        print('')
