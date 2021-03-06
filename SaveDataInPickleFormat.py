import pickle
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from collections import namedtuple
from multiprocessing import Pool
import sys

def CountStatistics(filename):
  print('reading file ' + filename)

  nExamples = 0
  totFeatures = 0
  with open(filename) as fin:
    line = fin.readline().split()
    while line:
      nExamples += 1
      assert(len(line)>=2)
      totFeatures += len(line)-2
      line = fin.readline().split()

  return nExamples, totFeatures



def ReadDataFromSVMLightFile(filename):
  print('reading file ' + filename)
  with open(filename) as fin:
    line = fin.readline().strip().split()
    nexamples = int(line[0])
    nfeatures = int(line[1])
    nlabels = int(line[2])
    x = lil_matrix((nexamples, nfeatures+1), dtype=np.float)
    y = lil_matrix((nexamples, nlabels+1), dtype=np.float)
    for i in range(nexamples):
      line = fin.readline().strip().split(',')
      if (len(line) > 1):
        for l in line[:-1]:
          y[i, int(l)] = 1
      features = line[-1].split()
      if ':' not in features[0]:
        y[i, int(features[0])] = 1
        features = features[1:]
      for f in features:
        try:
          fid, fval = f.split(':')
        except:
          print('error at: ' + str(i) + ' ' + f)
          exit()
        if (float(fval) != 0):
          x[i, int(fid)] = float(fval)
  # make sure the feature id start from zero
  if (x[:, 0].nnz == 0):
    x = x[:, 1:]
  else:
    assert(x[:, nfeatures].nnz == 0)
    x = x[:, :nfeatures]
  # also make sure the label id start from zero
  if (y[:, 0].nnz == 0):
    y = y[:, 1:]
  else:
    assert(y[:, nlabels].nnz == 0)
    y = y[:, :nlabels]
  # convert x and y into csr_matrix and return
  return csr_matrix(x), csr_matrix(y)


class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt


if __name__ == '__main__':
  dataFile = sys.argv[1]
  trainSplit = float(sys.argv[2])
  pickleFile = sys.argv[3]

  if pickleFile[-4:] != '.pkl':
    pickleFile = pickleFile + '.pkl'

  X, Y = ReadDataFromSVMLightFile(dataFile)
  numSamples = X.shape[0]
  trainSamples = int(numSamples*trainSplit)
  Xt, Yt = (X[trainSamples:, :], Y[trainSamples:, :])
  X, Y = (X[:trainSamples, :], Y[:trainSamples, :])
  
  data = Data(X = X, Y = Y, Xt = Xt, Yt = Yt)
  pickle.dump(data, open(pickleFile, 'wb'), pickle.HIGHEST_PROTOCOL)
