import pickle
from scipy.sparse import issparse, identity
import labelCount as lc
from sklearn.preprocessing import normalize
import numpy as np
import math


class Data:
  def __init__(self, X, Y, Xt, Yt):
    self.X = X
    self.Y = Y
    self.Xt = Xt
    self.Yt = Yt


def ComputeErrors(filename, data):
  res = pickle.load(open(filename, 'rb'))
  labelProjectionMatrix = res['labelProjMatrix']
  featureProjectionMatrix = res['featureProjMatrix']
  if (issparse(data.X)):
    projFeatureMatrixTrain = data.X * featureProjectionMatrix
    projFeatureMatrixTest = data.Xt * featureProjectionMatrix
  else:
    projFeatureMatrixTrain = np.matmul(data.X, featureProjectionMatrix)
    projFeatureMatrixTest = np.matmul(data.Xt, featureProjectionMatrix)
  projLabelMatrixTrain = data.Y * labelProjectionMatrix
  projLabelMatrixTest = data.Yt * labelProjectionMatrix
  errorsTrain = np.sum(np.maximum(0, 1-np.multiply(projFeatureMatrixTrain, projLabelMatrixTrain)), axis=0)
  errorsTest = np.sum(np.maximum(0, 1-np.multiply(projFeatureMatrixTest, projLabelMatrixTest)), axis=0)
  return errorsTrain, errorsTest


def ComputePerLabelMargin(filename, data):
  res = pickle.load(open(filename, 'rb'))
  labelProjectionMatrix = res['labelProjMatrix']
  featureProjectionMatrix = res['featureProjMatrix']
  if (issparse(data.X)):
    projFeatureMatrixTrain = data.X * featureProjectionMatrix
    projFeatureMatrixTest = data.Xt * featureProjectionMatrix
  else:
    projFeatureMatrixTrain = np.matmul(data.X, featureProjectionMatrix)
    projFeatureMatrixTest = np.matmul(data.Xt, featureProjectionMatrix)
  #marginTrainList = []
  #marginTestList = []
  meanMarginMatrix = np.zeros((featureProjectionMatrix.shape[1], data.Y.shape[1])).T
  stdMarginMatrix = np.zeros((featureProjectionMatrix.shape[1], data.Y.shape[1])).T
  skewnessMatrix = np.zeros((featureProjectionMatrix.shape[1], data.Y.shape[1])).T
  meanMarginMatrixTe = np.zeros((featureProjectionMatrix.shape[1], data.Y.shape[1])).T
  stdMarginMatrixTe = np.zeros((featureProjectionMatrix.shape[1], data.Y.shape[1])).T
  skewnessMatrixTe = np.zeros((featureProjectionMatrix.shape[1], data.Y.shape[1])).T
  for l in range(data.Y.shape[1]):
    labelMatrixTrain = np.zeros(data.Y.shape)+data.Y[:, l]
    labelMatrixTest = np.zeros(data.Yt.shape)+data.Yt[:, l]
    projLabelMatrixTrain = labelMatrixTrain * labelProjectionMatrix
    projLabelMatrixTest = labelMatrixTest * labelProjectionMatrix
    marginTrain = np.sort(np.multiply(projFeatureMatrixTrain, projLabelMatrixTrain), axis=0)
    marginTest = np.sort(np.multiply(projFeatureMatrixTest, projLabelMatrixTest), axis=0)
    #marginTrainList.append(marginTrain)
    #marginTestList.append(marginTest)
    nExamples = np.sum(data.Y[:, l])
    totExamples = data.Y.shape[0]
    meanMarginMatrix[l, :] = np.sum(marginTrain, axis=0)/nExamples
    stdMarginMatrix[l, :] = np.std(marginTrain, axis=0)*math.sqrt(float(totExamples)/nExamples)
    skewnessMatrix[l, :] = np.sum(marginTrain>0, axis=0)/float(nExamples)

    nExamplesTe = np.sum(data.Yt[:, l])
    totExamplesTe = data.Yt.shape[0]
    meanMarginMatrixTe[l, :] = np.sum(marginTest, axis=0)/nExamplesTe
    stdMarginMatrixTe[l, :] = np.std(marginTest, axis=0)*math.sqrt(float(totExamplesTe)/nExamplesTe)
    skewnessMatrixTe[l, :] = np.sum(marginTest>0, axis=0)/float(nExamplesTe)

  res['meanMarginMatrix'] = meanMarginMatrix
  res['stdMarginMatrix'] = stdMarginMatrix
  res['skewnessMatrix'] = skewnessMatrix
  res['meanMarginMatrixTe'] = meanMarginMatrixTe
  res['stdMarginMatrixTe'] = stdMarginMatrixTe
  res['skewnessMatrixTe'] = skewnessMatrixTe
  return res


def LoadData(i):
  labelStruct = lc.labelStructs[i]
  dataFile = labelStruct.fileName
  data = pickle.load(open(dataFile, 'rb'))
  data.X = normalize(data.X, norm='l2', axis=1)
  data.Xt = normalize(data.Xt, norm='l2', axis=1)
  return data


if __name__ == '__main__':
  data = LoadData(5)
  resFile = 'Results/MOBCAP_res_aloi_TS0_MU110000_MU20.0001_MU30.01_MU40_D15_IT15.pkl'
  trainErrors, testErrors = ComputeErrors(resFile, data)
  marginRes = ComputePerLabelMargin(resFile, data)
  marginRes['trainErrors'] = trainErrors
  marginRes['testErrors'] = testErrors
  pickle.dump(marginRes, open('MarginResult.pkl', 'wb'))
