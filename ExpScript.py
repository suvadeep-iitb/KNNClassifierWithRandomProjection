#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import labelCount as lc
from RandomEmbeddingAKNNPredictor import RandomEmbeddingAKNNPredictor as KNNPredictor


#Data = namedtuple("Data", "X Y Xt Yt")
class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt

params = {
  "numLearners": 1, # Currently works for only 1
  "numThreads": 3,
  "embDim": 100,
  "normalize": 1,
  "lamb": 1,
  "maxTestSamples": 2000000,
  "maxTrainSamples": 20000000}

lambdaList = [0.1]
nnTestList = [3, 5, 10]
embDimList = [100]

for i in [2]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print("Running for " + dataFile)
  data = pickle.load(open(dataFile, 'rb'))
  # For related search data, feature matrix in dense

  # Perform initial random permutation of the data
  print("Random permuting the data ...")
  perm = np.random.permutation(data.X.shape[0])
  data.X = data.X[perm, :]
  data.Y = data.Y[perm, :]
  perm = np.random.permutation(data.Xt.shape[0])
  data.Xt = data.Xt[perm, :]
  data.Yt = data.Yt[perm, :]

  params["featureDim"] = data.X.shape[1]
  params["labelDim"] = data.Y.shape[1]

  if params["normalize"] == 1:
    print("Normalizing data ...")
    data.X = normalize(data.X, norm = 'l2', copy = True);
    data.Xt = normalize(data.Xt, norm = 'l2', copy = True);
    print("Normalization done")

  resFilePrefix = labelStruct.resFile;
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
                         params['numThreads'])
      trainResList = knnPredictor.PredictAndComputePrecision(
                         data.X,
                         data.Y,
                         nnTestList,
                         params['maxTestSamples'],
                         params['numThreads'])
      resFile = 'Results/'+resFilePrefix+'_L'+str(ed)+'.pkl'
      pickle.dump((testResList, trainResList, nnTestList), open(resFile, 'wb'))
      print('')
      print('')
      print('')
