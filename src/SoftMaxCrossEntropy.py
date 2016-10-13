import numpy as np

class SoftMaxCrossEntropy(object):

    def __init__(self, layerType):
        self.layerType = layerType
        self.previousLayer = None
        self.lossOutput = None
        self.backwardOutput = None

    def LossOutput(self, target):
        self.lossOutput = np.sum(self.previousLayer.forwardOutput - target)

    def BackwardOutput(self, target):
        self.backwardOutput =  self.previousLayer.forwardOutput - target
