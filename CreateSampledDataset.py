import sys
import pickle
import numpy as np
from scipy.sparse import csr_matrix

class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt


def CreateSampledDataset(data, sampleSize):
  [n, d] = data.X.shape
  [nt, d] = data.Xt.shape
  [n, l] = data.Y.shape

  np.random.seed(1)

  # Sample labels
  perm = np.random.permutation(l)[:sampleSize]
  Ysam = data.Y[:, perm]
  Ytsam = data.Yt[:, perm]

  print(str(Ysam.shape))
  print(str(Ytsam.shape))
  print(str(perm.shape))

  # Remove examples from the train data zero active labels
  labelCountPerExamples = Ysam.getnnz(1)
  X = data.X[labelCountPerExamples>0, :]
  Y = Ysam[labelCountPerExamples>0, :]

  # Remove examples from the test data zero active labels
  labelCountPerExamples = Ytsam.getnnz(1)
  Xt = data.Xt[labelCountPerExamples>0, :]
  Yt = Ytsam[labelCountPerExamples>0, :]

  data.X = X
  data.Y = Y
  data.Xt = Xt
  data.Yt = Yt

  assert(data.X.shape[0] == data.Y.shape[0])
  assert(data.Xt.shape[0] == data.Yt.shape[0])
  assert(data.Y.shape[1] == data.Yt.shape[1])

  print("Active labels: "+str(data.Y.shape[1]))
  print("New trainset size: "+str(data.X.shape[0]))
  print("New testset size: "+str(data.Xt.shape[0]))

  return data


if __name__ == '__main__':
  inputfile = sys.argv[1]
  outputfile = sys.argv[2]
  sampleSize = int(sys.argv[3])

  data = pickle.load(open(inputfile, 'rb'))
  data = CreateSampledDataset(data, sampleSize)
  pickle.dump(data, open(outputfile, 'wb'))
