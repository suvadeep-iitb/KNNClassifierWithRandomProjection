import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
from scipy.optimize import minimize
import pickle
import nmslib
import math
from liblinearutil import *
from datetime import datetime
from RandomEmbeddingAKNNPredictor import *
from numpy.linalg import norm
from scipy.optimize import minimize
from scipy.stats import truncnorm
from datetime import datetime
#import tensorflow as tf


class MultipleOrthogonalBinaryAKNNPredictor(RandomEmbeddingAKNNPredictor):

  def __init__(self, params):
    self.embDim = params['embDim']
    self.mu1 = params['mu1']
    self.mu2 = params['mu2']
    self.mu3 = params['mu3']
    self.featureDim = params['featureDim']
    self.labelDim = params['labelDim']
    self.batchSize = params['batchSize']
    self.maxTrainSamples = 0
    self.seed = params['seed']
    self.isSparse = params['isSparse']
    self.maxActiveFeatures = params['maxActiveFeatures']
    self.maxActiveLabels = params['maxActiveLabels']
    self.innerIter = params['innerIter']
    self.paramSaveFile = params['paramSaveFile']
    self.outerIter = 0
    self.maxTrainSample = 0
    self.sampleIndices = []

  
  def SaveParams(self, params, paramSaveFile):
    if (self.paramSaveFile):
      pickle.dump(params, 
                  open(paramSaveFile, 'wb'), 
                  pickle.HIGHEST_PROTOCOL)


  def LoadParams(self, paramSaveFile):
    return pickle.load(open(paramSaveFile, 'rb'))


  def LearnParams(self, X, Y, itr=1):
    featureProjMatrix = GenerateInitialFeatureProjectionMatrix(self.featureDim, self.embDim)
    labelProjMatrix = GenerateInitialLabelProjectionMatrix(self.labelDim, self.embDim)
    log = ""
    for i in range(itr):
      resFeatureOpt = self.LearnFeatureProjMatrix(X, Y, featureProjMatrix)
      featureProjMatrix = np.reshape(resFeatureOpt.x, (X.shape[1], -1))
      objValue_W = resFeatureOpt.fun
      print(str(datetime.now()) + " : " + "Iter=" + str(i+1) + " objValue_W="+str(self.objValue_W))
      log += str(datetime.now()) + " : " + "Iter=" + str(i+1) + " objValue_W="+str(self.objValue_W) + "\n"

      resLabelOpt = self.LearnLabelProjMatrix(X, Y, labelProjMatrix)
      labelProjMatrix = np.reshape(resLabelOpt.x, (Y.shape[1], -1))
      objValue_F = resLabelOpt.fun
      print(str(datetime.now()) + " : " + "Iter=" + str(i+1) "objValue_F="+str(self.objValue_F))
      log += str(datetime.now()) + " : " + "Iter=" + str(i+1) "objValue_F="+str(self.objValue_F) + "\n"

      self.outerIter += 1
      self.SaveParams((featureProjMatrix, labelProjMatrix, objValue_W, objValue_F, log), self.paramSaveFile)

    return featureProjMatrix, labelProjMatrix, objValue_W, objValue_F, log


  def LearnFeatureProjMatrix(self, X, Y):
    assert(issparse(Y))
    projLabelMatrix = Y*self.labelProjMatrix
    return minimize(fun=objFunction_W, 
                    x0=self.featureProjMatrix.flatten(), 
                    args=(X, projLabelMatrix, self.mu3), 
                    method='BFGS', 
                    jac=True, 
                    options={'maxiter': self.innerIter, 'disp': True})

    
  def LearnLabelProjMatrix(self, X, Y):
    assert(issparse(Y))
    if (issparse(X)):
      projFeatureMatrix = X*W
    else:
      projFeatureMatrix = np.matmul(X, W)
    bounds = [(-1, 1)]*self.labelProjMatrix.size
    return minimize(fun=self.objFunction_F, 
                    x0=self.featureProjMatrix.flatten(), 
                    args=(Y, projFeatureMatrix, self.mu1, self.mu2)
                    method='BFGS', 
                    bounds=bounds, 
                    jac=True, 
                    options={'maxiter': self.innerIter, 'disp': True})


  def objFunction_W(x, (X, projLabelMatrix, mu3)):
    if (issparse):
      margin = np.multiply((X*np.reshape(x, (X.shape[1], -1))), projLabelMatrix) - 1
      grad = np.transpose(X) * np.multiply(projLabelMatrix, (margin > 0))
    else:
      margin = np.multiply(np.matmul(X, np.reshape(x, (X.shape[1], -1))), projLabelMatrix) - 1
      grad = np.matmul(np.transpose(X), np.multiply(projLabelMatrix, (margin > 0)))
    objVal = np.sum(np.maximum(margin, 0)) + mu3 * (norm(x)**2)
    grad = grad.flatten() + 2 * mu3 * x
    return objVal, grad


  def objFunction_F(x, (Y, projFeatureMatrix, mu1, mu2)):
    x = np.reshape(x, (Y.shape[1], -1))
    margin = np.multiply((Y*x), projFeatureMatrix) - 1
    crossProd = np.matmul(np.transpose(x), x)
    np.fill_diagonal(crossProd, 0)
    objVal = np.sum(np.maximum(margin, 0)) 
             + mu1 * norm(np.sum(x, axis=0), 1)
             + mu2 * np.sum(crossProd)
    grad = np.transpose(Y) * np.multiply(projFeatureMatrix, (matgin > 0))
           + mu1 * np.ones((x.shape[0], 1)) * sign(np.sum(x, axis=0))
           + mu2 * 2 * x * sign(crossProd)
    return objVal, grad.flatten()

 
  def GenerateInitialFeatureProjectionMatrix(nrow, ncol):
    return np.random.randn(nrow, ncol)/math.sqrt(nrow)


  def GenerateInitialLabelProjectionMatrix(nrow, ncol):
    return np.reshape(trunnorm.rvs(-1, 1, size=nrow*ncol), (nrow, ncol))
