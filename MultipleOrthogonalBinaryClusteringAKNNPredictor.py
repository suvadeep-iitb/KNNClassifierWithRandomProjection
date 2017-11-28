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
#import tensorflow as tf


class MultipleOrthogonalBinaryClusteringAKNNPredictor(RandomEmbeddingAKNNPredictor):

  def __init__(self, params):
    self.embDim = params['embDim']
    self.mu1 = params['mu1']
    self.mu2 = params['mu2']
    self.mu3 = params['mu3']
    self.featureDim = params['featureDim']
    self.labelDim = params['labelDim']
    #self.batchSize = params['batchSize']
    self.seed = params['seed']
    #self.isSparse = params['isSparse']
    #self.maxActiveFeatures = params['maxActiveFeatures']
    #self.maxActiveLabels = params['maxActiveLabels']
    self.innerIter = params['innerIter']
    self.logFile = params['logFile']
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
    self.featureProjMatrix = GenerateInitialFeatureProjectionMatrix(self.featureDim, self.embDim, self.seed)
    self.labelProjMatrix = GenerateInitialLabelProjectionMatrix(self.labelDim, self.embDim, self.seed+1023)
    self.log = ""
    for i in range(itr):
      resLabelOpt = self.LearnLabelProjMatrix(X, Y, self.featureProjMatrix, self.labelProjMatrix)
      self.labelProjMatrix = np.reshape(resLabelOpt.x, (Y.shape[1], -1))
      self.objValue_F = resLabelOpt.fun
      print(str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_F = " + str(self.objValue_F))
      self.log += str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_F = " + str(self.objValue_F) + "\n"

      resFeatureOpt = self.LearnFeatureProjMatrix(X, Y, self.featureProjMatrix, self.labelProjMatrix)
      self.featureProjMatrix = np.reshape(resFeatureOpt.x, (X.shape[1], -1))
      self.objValue_W = resFeatureOpt.fun
      print(str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_W = " + str(self.objValue_W))
      self.log += str(datetime.now()) + " : " + "Iter = " + str(i+1) + " objValue_W = " + str(self.objValue_W) + "\n"

      self.outerIter += 1
      self.SaveParams((self.featureProjMatrix, self.labelProjMatrix, self.objValue_W, self.objValue_F, self.log), self.logFile)


  def LearnFeatureProjMatrix(self, X, Y, featureProjMatrix, labelProjMatrix):
    assert(issparse(Y))
    projLabelMatrix = Y*labelProjMatrix
    return minimize(fun=objFunction_W, 
                    x0=featureProjMatrix.flatten(), 
                    args=(X, projLabelMatrix, self.mu3), 
                    method='Newton-CG', 
                    jac=True, 
                    options={'maxiter': self.innerIter, 'disp': True})

    
  def LearnLabelProjMatrix(self, X, Y, featureProjMatrix, labelProjMatrix):
    assert(issparse(Y))
    if (issparse(X)):
      projFeatureMatrix = X*featureProjMatrix
    else:
      projFeatureMatrix = np.matmul(X, featureProjMatrix)
    bounds = [(-1, 1)]*labelProjMatrix.size
    return minimize(fun=objFunction_F, 
                    x0=labelProjMatrix.flatten(), 
                    args=(Y, projFeatureMatrix, self.mu1, self.mu2),
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    jac=True, 
                    options={'maxiter': self.innerIter, 'disp': True})


def GenerateInitialFeatureProjectionMatrix(nrow, ncol, seed):
  np.random.seed(seed)
  return np.random.randn(nrow, ncol)/math.sqrt(nrow)


def GenerateInitialLabelProjectionMatrix(nrow, ncol, seed):
  np.random.seed(seed)
  return np.reshape(truncnorm.rvs(-1, 1, size=nrow*ncol), (nrow, ncol))


def objFunction_W(x, X, projLabelMatrix, mu3):
  embDim = float(projLabelMatrix.shape[1])
  featureDim = float(X.shape[1])
  if (issparse):
    margin = 1 - np.multiply((X*np.reshape(x, (X.shape[1], -1))), projLabelMatrix)
    grad = - np.transpose(X) * np.multiply(projLabelMatrix, (margin > 0))
  else:
    margin = 1 - np.multiply(np.matmul(X, np.reshape(x, (X.shape[1], -1))), projLabelMatrix)
    grad = - np.matmul(np.transpose(X), np.multiply(projLabelMatrix, (margin > 0)))
  objVal = np.sum(np.maximum(margin, 0))/(X.shape[0]*embDim) \
           + (mu3/(embDim*featureDim)) * (norm(x)**2)
  grad = grad.flatten()/(X.shape[0]*embDim) + 2 * (mu3/(embDim*featureDim)) * x
  return objVal, grad


def objFunction_F(x, Y, projFeatureMatrix, mu1, mu2):
  embDim = float(projFeatureMatrix.shape[1])
  x = np.reshape(x, (Y.shape[1], -1))
  margin = 1 - np.multiply((Y*x), projFeatureMatrix)
  crossProd = np.matmul(np.transpose(x), x)
  np.fill_diagonal(crossProd, 0)
  objVal = np.sum(np.maximum(margin, 0))/(Y.shape[0]*embDim) \
           + (mu1/embDim) * norm(np.sum(x, axis=0), 2) \
           + (mu2/(embDim*(embDim-1))) * np.sum(np.power(crossProd, 2))
  grad = - np.transpose(Y) * np.multiply(projFeatureMatrix, (margin > 0))/(Y.shape[0]*embDim) \
         + (mu1/embDim) * 2 * np.matmul(np.ones((x.shape[0], 1)), (np.sum(x, axis=0, keepdims=True))) \
         + (mu2/(embDim*(embDim-1))) * 2 * 2 * np.matmul(x, (crossProd))
  return objVal, grad.flatten()

 

