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


if __name__ == '__main__':
  inputfile = sys.argv[1]
  outputfile = sys.argv[2]

  data = pickle.load(open(inputfile, 'rb'))
  print("# of features in original dataset: "+str(data.X.shape[1]))
  n = np.array(np.sum(np.abs(data.X), axis=0)).reshape(-1)
  data.X = data.X[:, n>0]
  data.Xt = data.Xt[:, n>0]
  print("# of features in new dataset: "+str(data.X.shape[1]))
  pickle.dump(data, open(outputfile, 'wb'))
