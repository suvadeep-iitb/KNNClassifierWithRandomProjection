import pickle
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from collections import namedtuple
import labelCount as lc
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



def ReadDataFromODPFile(filename):
  nExamples, totFeatures = CountStatistics(filename)
  print('# of examples: '+str(nExamples)+' # of features: '+str(totFeatures))

  indptr = np.zeros((nExamples+1,), dtype=np.int64)
  indices = np.zeros((totFeatures,), dtype=np.int32)
  data = np.zeros((totFeatures,), dtype=np.float)

  l_indptr = np.zeros((nExamples+1,), dtype=np.int64)
  l_indices = np.zeros((nExamples,), dtype=np.int32)
  l_data = np.zeros((nExamples,), dtype=np.float)

  with open(filename) as fin:
    indptr[0] = 0
    l_indptr[0] = 0
    for e in range(nExamples):
      line = fin.readline().split()
      l_indptr[e+1] = e+1
      l_indices[e] = int(line[0])
      l_data[e] = 1.0
      line = line[2:]
      indptr[e+1] = indptr[e]+len(line)
      for p in range(len(line)):
        fid, fval = line[p].split(':')
        indices[indptr[e]+p] = int(fid)
        data[indptr[e]+p] = float(fval)
  assert(indptr[nExamples] == totFeatures)

  x = csr_matrix((data, indices, indptr))
  y = csr_matrix((l_data, l_indices, l_indptr))

  return x, y


def ReadDataFromRelatedSearchFile(fileName):
  with open(fileName) as fin:
    line = fin.readline().strip().split()
    nRow = int(line[0])
    nCol = int(line[1])
    M = lil_matrix((nRow, nCol+1), dtype=np.float)
    for i in range(nRow):
      line = fin.readline().strip().split()
      for t in line:
        col, val = t.split(':')
        if (float(val) != 0):
          M[i, int(col)] = float(val)
  # Make sure the element id start from zero
  if (M[:, 0].nnz == 0):
    M = M[:, 1:]
  else:
    assert(M[:, nCol].nnz == 0)
    M = M[:, :nCol]
  return csr_matrix(M)


class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt


if __name__ == '__main__':
  trainFile = sys.argv[1]
  testFile = sys.argv[2]
  pickleFile = sys.argv[3]

  if pickleFile[-4:] != '.pkl':
    pickleFile = pickleFile + '.pkl'

  X, Y = ReadDataFromODPFile(trainFile)
  Xt, Yt = ReadDataFromODPFile(testFile)
  
  data = Data(X = X, Y = Y, Xt = Xt, Yt = Yt)
  pickle.dump(data, open(pickleFile, 'wb'), pickle.HIGHEST_PROTOCOL)
