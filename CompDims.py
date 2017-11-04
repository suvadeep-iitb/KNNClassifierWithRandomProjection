import pickle
import numpy as np
import labelCount as lc

class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt

for i in [8, 9, 10, 11]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print("Running for " + dataFile)
  data = pickle.load(open(dataFile, 'rb'))

  n, L = data.Y.shape
  nt, L = data.Yt.shape

  ppl = np.sum(data.Y, axis=0)
  avgPPL = np.mean(ppl)

  lpp = np.sum(data.Y, axis = 1)
  avgLPP = np.mean(lpp)
  maxLPP = np.max(lpp)

  print("# Train Point: "+str(n))
  print("# Test Point: "+str(nt))
  print("# Labels: "+str(L))
  print("Avg Points per Label: "+str(avgPPL))
  print("Avg Labels per Point: "+str(avgLPP))
  print("Max Labels per Point: "+str(maxLPP))
  print("")
  print("")
  print("")
