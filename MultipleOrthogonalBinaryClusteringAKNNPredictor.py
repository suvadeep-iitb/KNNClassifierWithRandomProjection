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
    self.batchSize = params['batchSize']
    self.maxTrainSamples = 0
    self.seed = params['seed']
    self.isSparse = params['isSparse']
    self.maxActiveFeatures = params['maxActiveFeatures']
    self.maxActiveLabels = params['maxActiveLabels']
    self.trainError = -1
    self.sampleIndices = []

    self.labelProjMatrix = tf.Variable(shape=[self.labelDim, self.embDim],
                                       initializer=tf.truncated_normal_initializer(stddev=0.5),
                                       trainable=True,
                                       name='LabelProjMatrix')
    self.featureProjMatrix = tf.Variable(shape=[self.featureDim, self.embDim],
                                         initializer=tf.truncated_normal_initializer(),
                                         trainable=True,
                                         name='FeatureProjMatrix')

    self.inputLabelVector = tf.Placeholder(shape=[self.batchSize, self.maxActiveLabels],
                                     dtype=tf.int64,
                                     name='InputLabelPlaceholder')
    self.validLabelMusk = tf.Placeholder(shape=[self.batchSize, self.maxActiveLabels],
                                         dtype=tf.float32,
                                         name='ValidLabelMusk')

    activeLabels = tf.embedding_lookup(self.labelProjMatrix, self.inputLabelVector)
    self.projectedLabelVector = tf.reduced_sum(tf.multiply(activeLabels, tf.expand_dims(self.validLabelMusk, axis=-1)), axis=1)

    if(self.isSparse):
      self.inputFeatureValue = tf.Placeholder(shape=[self.batchSize, self.maxActiveFeatures],
                                              dtype=tf.float32,
                                              name='InputFeatureValuePlaceholder')
      self.inputFeatureIndex = tf.Placeholder(shape=[self.batchSize, self.maxActiveFeatures],
                                              dtype=tf.int64,
                                              name='InputFeatureIndexPlaceholder')
      activeFeatures = tf.embedding_lookup(self.featureProjMatrix, self.inputFeatureIndex)
      self.projectedFeatureVector = tf.reduced_sum(tf.multiply(activeFeatures, tf.expand_dims(self.inputFeatureValue, axis=-1)), axis=1)
    else:
      self.inputFeatureVector = tf.Placeholder(shape=[self.batchSize, self.featureDim], 
                                             dtye=tf.float32,
                                             name='InputFeaturePlaceholder')
      self.projectedFeatureVector = tf.matmul(self.inputFeatureVector, self.featureProjMatrix)


