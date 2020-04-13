#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack, issparse
from sklearn.preprocessing import normalize
#import labelCount as lc
#from AKNNPredictor import AKNNPredictor as KNNPredictor
from RandomProjAKNNPredictor import RandomProjAKNNPredictor as KNNPredictor
from EnsembleAKNNPredictor import EnsembleAKNNPredictor



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
  "numLearners": 3,
  "numThreads": 20,
  "normalization": 'l2_row', # l2_row / l2_col / l1_row / l1_col / max_row / max_col
  "lamb": 1,
  "nnTestList": [1, 3, 10],
  "embDim": 20,
  "seed": 1,
  "logFile": '',
  "maxTestSamples": 0}

dataFile = "bibtex.pkl"

data = pickle.load(open(dataFile, 'rb'))

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

print("")
print("Hyper-parameters:")
print("\tData File               : %s" % (dataFile))
print("\tNum Learners            : %s" % (params["numLearners"]))
print("\tNum Threads             : %s" % (params["numThreads"]))
print("\tFeature Normalization   : %s" % (params["normalization"]))
print("\tLambda                  : %s" % (params["lamb"]))
print("\tK List                  : %s" % (params["nnTestList"]))
print("\tEmbedding Dimension     : %s" % (params["embDim"]))
print("\tSeed                    : %s" % (params["seed"]))
print("\tMax Test Samples        : %s" % (params["maxTestSamples"]))
print("")
print("")

nnTestList = params["nnTestList"]
params["basePredictor"] = KNNPredictor(params)
knnPredictor = EnsembleAKNNPredictor(params)
knnPredictor.Train(data.X, 
                   data.Y,
                   numThreads = params['numThreads'])
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

print("")
for (k, testRes, trainRes) in zip(nnTestList, testResList, trainResList):
    print("K: %s\tPrec \t Train \t\t Test" % (k))
    print("     \t1    \t %0.4f \t %0.4f" % (trainRes["precision"][0][0], testRes["precision"][0][0]))
    print("     \t2    \t %0.4f \t %0.4f" % (trainRes["precision"][1][0], testRes["precision"][1][0]))
    print("     \t3    \t %0.4f \t %0.4f" % (trainRes["precision"][2][0], testRes["precision"][2][0]))
    print("     \t4    \t %0.4f \t %0.4f" % (trainRes["precision"][3][0], testRes["precision"][3][0]))
    print("     \t5    \t %0.4f \t %0.4f" % (trainRes["precision"][4][0], testRes["precision"][4][0]))
    print("")
print("")
