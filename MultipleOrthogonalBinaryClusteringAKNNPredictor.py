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
    self.trainError = -1
    self.sampleIndices = []

  def LearnParams(self, X, Y):
    if (issparse(X)):
      self.objFunction = lambda W, F: np.sum(np.amax(np.multiply((X*W),(Y*F)) - 1, 0))
                                      + self.mu1 * norm(np.sum(F, axis=0), 1)
                                      + self.mu2 * GetOrthogonalityConstraint(F)
                                      + self.mu3 * (norm(W)**2)
    else:
      self.objFunction = lambda W, F: np.sum(np.amax(np.multiply(np.matmul(X,W),np.matmul(Y,F)) - 1, 0))
                                      + self.mu1 * norm(np.sum(F, axis=0), 1)
                                      + self.mu2 * GetOrthogonalityConstraint(F)
                                      + self.mu3 * (norm(W)**2)

    W0 = GenerateInitialFeatureProjectionMatrix(self.featureDim, self.embDim)
    F0 = GenerateInitialLabelProjectionMatrix(self.labelDim, self.embDim)

  def GetOrthogonalityConstraint(F):
    FTF = np.matmul(np.transpose(F), F)
    np.fill_diagonal(FTF, 0)
    return np.sum(FTF)
