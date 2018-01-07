from KNNPredictor import KNNPredictor
import numpy as np

class KNNPredictorOnLabelSpace(KNNPredictor):

  def __init__(self, params):
    pass


  def Train(self, X, Y, maxTrainSamples, numThreads):
    self.featureDim = X.shape[1]
    self.labelDim = Y.shape[1]
    return super(KNNPredictorOnLabelSpace, self).Train(Y, Y, maxTrainSamples, numThreads)


  def Predict(self, Yt, nnTest, numThreads = 1):
    assert(self.labelDim == Yt.shape[1])
    return super(KNNPredictorOnLabelSpace, self).Predict(Yt, nnTest, numThreads)


  def PredictAndComputePrecision(self, Xt, Yt, nnTestList, maxTestSamples, numThreads):
    assert(self.labelDim == Yt.shape[1])
    return super(KNNPredictorOnLabelSpace, self).PredictAndComputePrecision(Yt, Yt, nnTestList, maxTestSamples, numThreads)


  def ComputeKNN(self, Yt, nnTest, numThreads = 1):
    assert(self.labelDim == Yt.shape[1])
    return super(KNNPredictorOnLabelSpace, self).ComputeKNN(Yt, nnTest, numThreads)
