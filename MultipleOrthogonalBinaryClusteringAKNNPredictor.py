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
    self.featureProjMatrix = GenerateInitialFeatureProjectionMatrix(self.featureDim, self.embDim)
    self.labelProjMatrix = GenerateInitialLabelProjectionMatrix(self.labelDim, self.embDim)

  
  def SaveParams(self):
    if (self.paramSaveFile):
      pickle.dump((self.featureProjMatrix, self.labelProjMatrix), 
                  open(self.paramSaveFile, 'wb'), 
                  pickle.HIGHEST_PROTOCOL)


  def LoadParams(self):
    self.featureProjMatrix, self.labelProjMatrix = pickle.load(open(self.paramSaveFile, 'rb'))


  def LearnParams(self, X, Y, itr=1):
    for i in range(itr):
      resFeatureOpt = self.LearnFeatureProjMatrix(X, Y)
      self.featureProjMatrix = np.reshape(resFeatureOpt.x, (X.shape[1], -1))
      self.objValue_W = resFeatureOpt.fun
      print(str(datetime.now()) + " : " + "objValue_W="+str(self.objValue_W))

      resLabelOpt = self.LearnLabelProjMatrix(X, Y)
      self.labelProjMatrix = np.reshape(resLabelOpt.x, (Y.shape[1], -1))
      self.objValue_F = resLabelOpt.fun
      print(str(datetime.now()) + " : " + "objValue_F="+str(self.objValue_F))

      self.outerIter += 1
      self.SaveParams()


  def LearnFeatureProjMatrix(self, X, Y):
    assert(issparse(Y))
    projLabelMatrix = Y*self.labelProjMatrix
    if (issparse(X)):
      self.objFunction_W = lambda W: np.sum(np.maximum(np.multiply((X*np.reshape(W, (X.shape[1], -1))), projLabelMatrix) - 1, 0.0))
                                     + self.mu3 * (norm(W)**2)
    else:
      self.objFunction_W = lambda W: np.sum(np.maximum(np.multiply(np.matmul(X,np.reshape(W, (X.shape[1], -1))), projLabelMatrix) - 1, 0.0))
                                     + self.mu3 * (norm(W)**2)
    return minimize(self.objFunction_W, self.featureProjMatrix, method='BFGS', jac=False, options={'maxiter': self.innerIter, 'disp': True})

    
 def LearnLabelProjMatrix(self, X, Y):
    assert(issparse(Y))
    if (issparse(X)):
      projFeatureMatrix = X*W
    else:
      projFeatureMatrix = np.matmul(X, W)
    self.objFunction_F = lambda F: np.sum(np.maximum(np.multiply(projFeatureMatrix, (Y*np.reshape(F, (Y.shape[1], -1)))) - 1, 0.0))
                                   + self.mu1 * norm(np.sum(np.reshape(F, (Y.shape[1], -1)), axis=0), 1)
                                   + self.mu2 * GetOrthogonalityConstraint(np.reshape(F, (Y.shape[1], -1)))
    bounds = [(-1, 1)]*self.labelProjMatrix.size
    return minimize(self.objFunction_F, self.featureProjMatrix, method='BFGS', bounds=bounds, jac=False, options={'maxiter': self.innerIter, 'disp': True})


  def objFunction_W(x, (X, projLabelMatrix, mu)):
    if (issparse):
      margin = np.multiply((X*np.reshape(x, (X.shape[1], -1))), projLabelMatrix) - 1
      grad = np.transpose(X) * np.multiply(projLabelMatrix, margin > 0)
    else:
      margin = np.multiply(np.matmul(X, np.reshape(x, (X.shape[1], -1))), projLabelMatrix) - 1
      grad = np.matmul(np.transpose(X), np.multiply(projLabelMatrix, margin > 0))
    objVal = np.sum(np.maximum(margin, 0)) + mu * (norm(x)**2)
    return objVal, grad

 
  def GetOrthogonalityConstraint(F):
    FTF = np.matmul(np.transpose(F), F)
    np.fill_diagonal(FTF, 0)
    return np.sum(FTF)


  def GenerateInitialFeatureProjectionMatrix(nrow, ncol):
    return np.random.randn(nrow, ncol)/m.sqrt(nrow)


  def GenerateInitialLabelProjectionMatrix(nrow, ncol):
    return np.reshape(trunnorm.rvs(-1, 1, size=nrow*ncol), (nrow, ncol))
