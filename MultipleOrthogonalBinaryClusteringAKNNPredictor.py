import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, vstack, issparse
import pickle
import nmslib
import math
from liblinearutil import *
from datetime import datetime
from RandomEmbeddingAKNNPredictor import *
import tensorflow as tf


class MultipleOrthogonalBinaryAKNNPredictor(RandomEmbeddingAKNNPredictor):

  def __init__(self, params):
    self.embDim = params['embDim']
    self.mu1 = params['mu1']
    self.mu2 = params['mu2']
    self.mu3 = params['mu3']
    self.featureDim = params['featureDim']
    self.labelDim = params['labelDim']
    self.maxTrainSamples = 0
    self.trainError = -1
    self.sampleIndices = []

    self.labelProjMatrix = tf.Variable([self.labelDim, self.embDim])
