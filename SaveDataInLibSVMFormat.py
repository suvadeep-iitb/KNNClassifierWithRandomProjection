import pickle
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from collections import namedtuple
import labelCount as lc
from multiprocessing import Pool



def WriteDataInSVMLightFormat(X, Y, fileName):
  print('  Writing into file ' + fileName)
  nExamples = X.shape[0]
  nFeatures = X.shape[1]
  nLabels = Y.shape[1]
  with open(fileName, 'w') as fin:
    fin.write(str(nExamples)+' '+str(nFeatures)+' '+str(nLabels)+'\n')
    for i in range(nExamples):
      (rows, cols) = Y[i, :].nonzero()
      fin.write(str(cols[0]))
      for c in cols[1:]:
        fin.write(','+str(c))
      (rows, cols) = X[i, :].nonzero()
      for c in cols:
        fin.write(' '+str(c)+':'+str(X[i, c]))
      fin.write('\n')


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



#Data = namedtuple("Data", "X Y Xt Yt")
class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt


for i in [6]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print('Running for ' + dataFile)

  filePrefix = dataFile[:-4]
  trainFile = filePrefix + '_train.txt'
  testFile = filePrefix + '_100K_test.txt'

  data = pickle.load(open(dataFile, 'rb'))
  perm = np.random.permutation(data.Xt.shape[0])
  data.Xt = data.Xt[perm[:100000], :]
  data.Yt = data.Yt[perm[:100000], :]
  #WriteDataInSVMLightFormat(csr_matrix(data.X), data.Y, trainFile)
  WriteDataInSVMLightFormat(csr_matrix(data.Xt), data.Yt, testFile)