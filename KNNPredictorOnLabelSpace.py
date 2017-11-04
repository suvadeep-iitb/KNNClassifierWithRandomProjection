from KNNPredictor import KNNPredictor
import numpy as np

class KNNPredictorOnLabelSpace(KNNPredictor):

  def __init__(self, params):
    self.featureDim = params['labelDim']
    self.labelDim = params['labelDim']


  def Train(self, X, Y, maxTrainSamples, numThreads):
    return super(KNNPredictorOnLabelSpace, self).Train(Y, Y, maxTrainSamples, numThreads)


  def Predict(self, Yt, nnTest, numThreads = 1):
    return super(KNNPredictorOnLabelSpace, self).Predict(Yt, nnTest, numThreads)


  def PredictAndComputePrecision(self, Xt, Yt, nnTestList, maxTestSamples, numThreads):
    return super(KNNPredictorOnLabelSpace, self).PredictAndComputePrecision(Yt, Yt, nnTestList, maxTestSamples, numThreads)


  def ComputeKNN(self, Yt, nnTest, numThreads = 1):
    return super(KNNPredictorOnLabelSpace, self).ComputeKNN(Yt, nnTest, numThreads)
