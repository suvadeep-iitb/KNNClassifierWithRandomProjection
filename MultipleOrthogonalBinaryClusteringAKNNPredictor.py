import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
from scipy.optimize import minimize
import pickle
import nmslib
import math
#from liblinearutil import *
from datetime import datetime
from RandomEmbeddingAKNNPredictor import *
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.stats import truncnorm
from datetime import datetime
from sklearn.preprocessing import scale, normalize
import os
#import tensorflow as tf


class MultipleOrthogonalBinaryClusteringAKNNPredictor(RandomEmbeddingAKNNPredictor):

  def __init__(self, params):
    self.embDim = params['embDim']
    self.mu1 = params['mu1']
    self.mu2 = params['mu2']
    self.mu3 = params['mu3']
    self.mu4 = params['mu4']
    self.seed = params['seed']
    self.innerIter = params['innerIter']
    self.logFile = params['logFile']
    self.initialFeatureProjMatrixFile = params['initialFeatureProjMatrixFile']
    self.outerIter = 0
    self.maxTrainSample = 0
    self.sampleIndices = []

  
  def SaveParams(self, params, paramSaveFile):
    if (paramSaveFile):
      pickle.dump(params, 
                  open(paramSaveFile, 'wb'), 
                  pickle.HIGHEST_PROTOCOL)


  def LoadParams(self, paramSaveFile):
    return pickle.load(open(paramSaveFile, 'rb'))


  def LearnParams(self, X, Y, itr=1, numThreads=1):
    featureDim = X.shape[1]
    self.featureProjMatrix = GenerateInitialFeatureProjectionMatrix(featureDim, self.embDim, self.initialFeatureProjMatrixFile, self.seed)
    self.labelProjMatrix = GenerateInitialLabelProjectionMatrix(X, Y, self.embDim, self.seed+1023)
    self.log = ""
    for i in range(itr):
      resLabelOpt = self.LearnLabelProjMatrix(X, Y, self.featureProjMatrix, self.labelProjMatrix)
      self.labelProjMatrix = np.reshape(resLabelOpt.x, (Y.shape[1], -1))
      self.objValue_F = resLabelOpt.fun
      self.grad_F = resLabelOpt.jac
      print(str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_F = " + str(self.objValue_F))
      self.log += str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_F = " + str(self.objValue_F) + "\n"

      resFeatureOpt = self.LearnFeatureProjMatrix(X, Y, self.featureProjMatrix, self.labelProjMatrix)
      self.featureProjMatrix = np.reshape(resFeatureOpt.x, (X.shape[1], -1))
      self.objValue_W = resFeatureOpt.fun
      self.grad_W = resFeatureOpt.jac
      print(str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_W = " + str(self.objValue_W))
      self.log += str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_W = " + str(self.objValue_W) + "\n"

      self.outerIter += 1
      self.SaveParams((self.featureProjMatrix, self.labelProjMatrix, self.objValue_W, self.objValue_F, self.grad_W, self.grad_F, self.log), self.logFile)


  def EmbedFeature(self, X, numThreads=1):
    if (issparse(X)):
      pX = (X * self.featureProjMatrix)
    else:
      pX = (np.matmul(X, self.featureProjMatrix))
    return pX


  def LearnFeatureProjMatrix(self, X, Y, featureProjMatrix, labelProjMatrix):
    assert(issparse(Y))
    projLabelMatrix = Y*labelProjMatrix
    return minimize(fun=objFunction_W, 
                    x0=featureProjMatrix.flatten(), 
                    args=(X, projLabelMatrix, self.mu3), 
                    method='Newton-CG', 
                    jac=True, 
                    options={'maxiter': self.innerIter, 'disp': False})

 
  def LearnLabelProjMatrix(self, X, Y, featureProjMatrix, labelProjMatrix):
    assert(issparse(Y))
    if (issparse(X)):
      projFeatureMatrix = X*featureProjMatrix
    else:
      projFeatureMatrix = np.matmul(X, featureProjMatrix)
    bounds = [(-1, 1)]*labelProjMatrix.size
    return minimize(fun=objFunction_F, 
                    x0=labelProjMatrix.flatten(), 
                    args=(Y, projFeatureMatrix, self.mu1, self.mu2, self.mu4),
                    method='TNC', 
                    bounds=bounds, 
                    jac=True, 
                    options={'maxiter': self.innerIter, 'disp': False})



def GenerateInitialFeatureProjectionMatrix(nrow, ncol, filename, seed):
  np.random.seed(seed)
  if filename:
    OvRWeightMatrix = pickle.load(open(filename, 'rb'))
    OvRWeightMatrix = normalize(OvRWeightMatrix, norm='l2', axis=0)
    assert(OvRWeightMatrix.shape[0] == nrow)
    R = np.random.randn(OvRWeightMatrix.shape[1], ncol)
    R[R>0] = 1.0
    R[R<0] = -1.0
    return np.matmul(OvRWeightMatrix, R)
  else:
    return np.random.randn(nrow, ncol)/math.sqrt(nrow)
  


def GenerateInitialLabelProjectionMatrix(X, Y, embDim, seed):
  np.random.seed(seed)
  nrow = Y.shape[1]
  ncol = embDim
  thr = 3
  return (np.reshape(truncnorm.rvs(-thr, thr, size=nrow*ncol), (nrow, ncol)))/thr


'''
def GenerateInitialLabelProjectionMatrix(X, Y, embDim, seed):
  filename = 'aloi_lpm_ED'+str(embDim)+'_S'+str(seed)+'.pkl'
  if (os.path.isfile(filename)):
    print('Loading initial labelProjMatrix from file')
    labelProjMatrix = pickle.load(open(filename, 'rb'))
  else:
    print('labelProjMatrix file does not exist, creating a new one')
    labelProjMatrix = GenerateCorrelatedLabelProjectionMatrix(X, Y, embDim, seed)
    pickle.dump(labelProjMatrix, open(filename, 'wb'))
  #labelProjMatrix = labelProjMatrix
  labelProjMatrix[labelProjMatrix>1] = 1.0
  labelProjMatrix[labelProjMatrix<-1] = -1.0
  return labelProjMatrix


def GenerateCorrelatedLabelProjectionMatrix(X, Y, embDim, seed):
  np.random.seed(seed)
  labelDim = Y.shape[1]
  featureDim = X.shape[1]
  nExamples = X.shape[0]
  labelCounts = np.sum(Y, axis=0)
  if (issparse(X)):
    meanLabelFeature = lil_matrix((labelDim, featureDim))
  else:
    meanLabelFeature = np.zeros((labelDim, featureDim))
  
  for labelId in range(labelDim):
    if (labelCounts[0, labelId] > 0):
      meanLabelFeature[labelId, :] = np.sum(X[np.squeeze(np.array((Y[:, labelId]>0.5).todense())), :], axis=0)/float(labelCounts[0, labelId])
  if (issparse(meanLabelFeature)):
    meanLabelFeature = csr_matrix(meanLabelFeature)
  meanLabelFeature = normalize(meanLabelFeature)

  randProjMatrix = np.sign(np.random.randn(featureDim, embDim))
  if (issparse(meanLabelFeature)):
    labelProjMatrix = meanLabelFeature * randProjMatrix / math.sqrt(embDim)
  else:
    labelProjMatrix = np.matmul(meanLabelFeature, randProjMatrix) / math.sqrt(embDim)
  return labelProjMatrix
'''


'''
def objFunction_W(x, X, projLabelMatrix, mu3):
  embDim = float(projLabelMatrix.shape[1])
  featureDim = float(X.shape[1])
  x = np.reshape(x, (X.shape[1], -1))
  if (issparse(X)):
    errorMatrix = X*x - projLabelMatrix
    objVal = norm(errorMatrix)**2
    grad = 2 * X.T * errorMatrix
  else:
    errorMatrix = np.matmul(X, x) - projLabelMatrix
    objVal = norm(errorMatrix)**2
    grad = 2 * np.matmul(X.T, errorMatrix)
  objVal = objVal/(X.shape[0]*embDim) \
           + (mu3/(embDim*featureDim)) * (norm(x)**2)
  grad = grad/(X.shape[0]*embDim) + 2 * (mu3/(embDim*featureDim)) * x
  return objVal, grad.flatten()


def objFunction_F(y, Y, projFeatureMatrix, mu1, mu2, mu4):
  embDim = float(projFeatureMatrix.shape[1])
  labelDim = float(Y.shape[1])
  y = np.reshape(y, (Y.shape[1], -1))
  errorMatrix = Y*y - projFeatureMatrix
  crossProd = np.matmul(np.transpose(y), y)
  np.fill_diagonal(crossProd, 0)
  objVal = - norm(errorMatrix)**2 / (Y.shape[0]*embDim) \
           + (mu1/embDim) * norm(np.sum(y, axis=0), 2) \
           + (mu2/(embDim*(embDim-1))) * np.sum(np.power(crossProd, 2)) \
           - (mu4/(embDim*labelDim)) * (norm(y)**2)
  grad = - 2 * Y.T * errorMatrix / (Y.shape[0]*embDim) \
         + (mu1/embDim) * 2 * np.matmul(np.ones((y.shape[0], 1)), (np.sum(y, axis=0, keepdims=True))) \
         + (mu2/(embDim*(embDim-1))) * 2 * 2 * np.matmul(y, (crossProd)) \
         - (mu4/(embDim*labelDim)) * 2 * y
  return objVal, grad.flatten()
'''


def objFunction_W(x, X, projLabelMatrix, mu3):
  embDim = float(projLabelMatrix.shape[1])
  featureDim = float(X.shape[1])
  x = np.reshape(x, (X.shape[1], -1))
  if (issparse(X)):
    margin = 1 - np.multiply((X*x), projLabelMatrix)
    grad = - np.transpose(X) * np.multiply(projLabelMatrix, (margin > 0))
  else:
    margin = 1 - np.multiply(np.matmul(X, x), projLabelMatrix)
    grad = - np.matmul(np.transpose(X), np.multiply(projLabelMatrix, (margin > 0)))
  objVal = np.sum(np.maximum(margin, 0))/(X.shape[0]*embDim) \
           + (mu3/(embDim*featureDim)) * (norm(x)**2)
  grad = grad/(X.shape[0]*embDim) + 2 * (mu3/(embDim*featureDim)) * x
  return objVal, grad.flatten()



def objFunction_F(x, Y, projFeatureMatrix, mu1, mu2, mu4):
  embDim = float(projFeatureMatrix.shape[1])
  labelDim = float(Y.shape[1])
  x = np.reshape(x, (Y.shape[1], -1))
  margin = 1 - np.multiply((Y*x), projFeatureMatrix)
  crossProd = np.matmul(np.transpose(x), x)
  np.fill_diagonal(crossProd, 0)
  objVal = np.sum(np.maximum(margin, 0))/(Y.shape[0]*embDim) \
           + (mu1/embDim) * norm(np.sum(x, axis=0), 2) \
           + (mu2/(embDim*(embDim-1))) * np.sum(np.power(crossProd, 2)) \
           - (mu4/(embDim*labelDim)) * (norm(x)**2)
  grad = - np.transpose(Y) * np.multiply(projFeatureMatrix, (margin > 0))/(Y.shape[0]*embDim) \
         + (mu1/embDim) * 2 * np.matmul(np.ones((x.shape[0], 1)), (np.sum(x, axis=0, keepdims=True))) \
         + (mu2/(embDim*(embDim-1))) * 2 * 2 * np.matmul(x, (crossProd)) \
         - (mu4/(embDim*labelDim)) * 2 * x
  #temp = 10.0
  #grad[np.logical_and(grad>-temp, grad<temp)] = temp * np.sign(x[np.logical_and(grad>-temp, grad<temp)])
  return objVal, grad.flatten()



'''
def objFunction_W(x, X, projLabelMatrix, mu3):
  embDim = float(projLabelMatrix.shape[1])
  featureDim = float(X.shape[1])
  x = np.reshape(x, (X.shape[1], -1))
  if (issparse(X)):
    objVal = - np.sum(np.multiply(X*x, projLabelMatrix)) / (X.shape[0]*embDim)
    grad = - np.transpose(X) * projLabelMatrix / (X.shape[0]*embDim)
  else:
    objVal = - np.sum(np.multiply(np.matmul(X, x), projLabelMatrix)) / (X.shape[0]*embDim)
    grad = - np.matmul(np.transpose(X), projLabelMatrix) / (X.shape[0]*embDim)
  objVal += (mu3/(embDim*featureDim)) * (norm(x)**2)
  grad += 2 * (mu3/(embDim*featureDim)) * x
  return objVal, grad.flatten()


def objFunction_F(x, Y, projFeatureMatrix, mu1, mu2, mu4):
  embDim = float(projFeatureMatrix.shape[1])
  labelDim = float(Y.shape[1])
  x = np.reshape(x, (Y.shape[1], -1))
  crossProd = np.matmul(np.transpose(x), x)
  np.fill_diagonal(crossProd, 0)
  objVal = - np.sum(np.multiply(Y*x, projFeatureMatrix)) / (Y.shape[0]*embDim) \
           + (mu1/embDim) * norm(np.sum(x, axis=0), 2) \
           + (mu2/(embDim*(embDim-1))) * np.sum(np.power(crossProd, 2)) \
           - (mu4/(embDim*labelDim)) * (norm(x)**2)
  grad = - np.transpose(Y) * projFeatureMatrix / (Y.shape[0]*embDim) \
         + (mu1/embDim) * 2 * np.matmul(np.ones((x.shape[0], 1)), (np.sum(x, axis=0, keepdims=True))) \
         + (mu2/(embDim*(embDim-1))) * 2 * 2 * np.matmul(x, (crossProd)) \
         - (mu4/(embDim*labelDim)) * 2 * x
  #grad[np.logical_and(grad<0.5, grad>0)] = 0.5
  #grad[np.logical_and(grad>-0.5, grad<0)] = -0.5
  
  return objVal, grad.flatten()
'''
