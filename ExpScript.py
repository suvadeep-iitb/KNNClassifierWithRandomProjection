#from collections import namedtuple
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import labelCount as lc
from RandomProjKNNPredictor import *


#Data = namedtuple("Data", "X Y Xt Yt")
class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt

params = {
  "numLearners": 1, # Currently works for only 1
  "numThreads": 1,
  "embDim": 100,
  "normalize": 1,
  "lamb": 1,
  "maxTestSamples": 500000,
  "maxTrainSamples": 10000000}

lambdaList = [0.0001, 0.01, 1, 100, 1000]
nnTestList = [5, 10, 20]
embDimList = [100, 500]

for i in [6]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print("Running for " + dataFile)
  data = pickle.load(open(dataFile, 'rb'))

  if params["normalize"] == 1:
    print("Normalizing data ...")
    data.X = normalize(data.X, norm = 'l2', copy = True);
    data.Xt = normalize(data.Xt, norm = 'l2', copy = True);
    print("Normalization done")

  params["resFilePrefix"] = labelStruct.resFile;
  for lam in lambdaList:
    params["lamb"] = lam
    for ed in embDimList:
      params["embDim"] = ed
      print("\tRunning for " + "lambda = " + str(params["lamb"]) + " emb_dim = " + str(params["embDim"]));   
      RandomProjKNNPredictor(data.X, data.Y, data.Xt, data.Yt, params, nnTestList); 
