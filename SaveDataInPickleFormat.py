import pickle
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from collections import namedtuple
import labelCount as lc
from multiprocessing import Pool



def ReadDataFromSVMLightFile(fileName):
  print('  Reading file ' + fileName)
  with open(fileName) as fin:
    line = fin.readline().strip().split()
    nExamples = int(line[0])
    nFeatures = int(line[1])
    nLabels = int(line[2])
    X = lil_matrix((nExamples, nFeatures+1), dtype=np.float)
    Y = lil_matrix((nExamples, nLabels+1), dtype=np.float)
    for i in range(nExamples):
      line = fin.readline().strip().split(',')
      if (len(line) > 1):
        for l in line[:-1]:
          Y[i, int(l)] = 1
      features = line[-1].split()
      if ':' not in features[0]:
        Y[i, int(features[0])] = 1
        features = features[1:]
      for f in features:
        try:
          fId, fVal = f.split(':')
        except:
          print('Error at: ' + str(i) + ' ' + f)
          exit()
        if (float(fVal) != 0):
          X[i, int(fId)] = float(fVal)
  # Make sure the feature id start from zero
  if (X[:, 0].nnz == 0):
    X = X[:, 1:]
  else:
    assert(X[:, nFeatures].nnz == 0)
    X = X[:, :nFeatures]
  # Also make sure the label id start from zero
  if (Y[:, 0].nnz == 0):
    Y = Y[:, 1:]
  else:
    assert(Y[:, nLabels].nnz == 0)
    Y = Y[:, :nLabels]
  # convert X and Y into csr_matrix and return
  return csr_matrix(X), csr_matrix(Y)



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


for i in [18]:
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  print('Running for ' + dataFile)
  
  filePrefix = dataFile[:-4]
  trainFile = filePrefix + '_train.txt'
  testFile = filePrefix + '_test.txt'

  X, Y = ReadDataFromSVMLightFile(trainFile)
  Xt, Yt = ReadDataFromSVMLightFile(testFile)
  
  '''
  datafile = '../DataSets/Delicious/Delicious_data.txt'
  trSplit = '../DataSets/Delicious/delicious_trSplit.txt'
  tstSplit = '../DataSets/Delicious/delicious_tstSplit.txt'
  Xfull, Yfull = ReadDataFromSVMLightFile(datafile)
  lines = open(trSplit).readlines()
  lines = [l.strip().split() for l in lines]
  trIdx = [[int(l)-1 for l in line] for line in lines]

  lines = open(tstSplit).readlines()
  lines = [l.strip().split() for l in lines]
  tstIdx = [[int(l)-1 for l in line] for line in lines]

  for i in range(10):
    tr = []
    for l in trIdx:
      tr.append(l[i])
    X = Xfull[tr, :]
    Y = Yfull[tr, :]
    te = []
    for l in tstIdx:
      te.append(l[i])
    Xt = Xfull[te, :]
    Yt = Yfull[te, :]

    data = Data(X, Y, Xt, Yt)
    #outputFile = filePrefix + '.pkl'
    outputFile = '../DataSets/Delicious/delicious_'+str(i)+'.pkl'
    pickle.dump(data, open(outputFile, 'wb'), pickle.HIGHEST_PROTOCOL)
'''
'''
trainFtFile = '../DataSets/RelatedSearch/trn_ft_mat.txt'
trainLblFile = '../DataSets/RelatedSearch/trn_lbl_mat.txt'
testFtFile = '../DataSets/RelatedSearch/tst_ft_mat.txt'
testLblFile = '../DataSets/RelatedSearch/tst_lbl_mat.txt'

X = ReadDataFromRelatedSearchFile(trainFtFile)
print("Train feature file reading done!")
Y = ReadDataFromRelatedSearchFile(trainLblFile)
print("Train label file reading done!")
Xt = ReadDataFromRelatedSearchFile(testFtFile)
print("Test feature file reading done!")
Yt = ReadDataFromRelatedSearchFile(testLblFile)
print("Test label file reading done!")
'''

data = Data(X = X, Y = Y, Xt = Xt, Yt = Yt)
pickle.dump(data, open(dataFile, 'wb'), pickle.HIGHEST_PROTOCOL)
