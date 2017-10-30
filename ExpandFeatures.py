import pickle
import numpy as np
import sys



class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt



def ExpandFeatures(M):
  [n, d] = M.shape
  Mnew = np.ones((n, int(d*(d+1)/2)+d+1))
  Mnew[:, 1: d+1] = M
  for r in range(n):
    # Populate the cross product terms
    for i in range(d):
      for j in range(i, d):
        Mnew[r, d+1+i*d-int(i*(i-1)/2)+j-i] = M[r, i]*M[r, j]
  return Mnew



if __name__ == '__main__':
  infile = sys.argv[1]
  outfile = sys.argv[2]

  data = pickle.load(open(infile, 'rb'))
  data.X = ExpandFeatures(data.X)
  data.Xt = ExpandFeatures(data.Xt)

  pickle.dump(data, open(outfile, 'wb'), pickle.HIGHEST_PROTOCOL)
