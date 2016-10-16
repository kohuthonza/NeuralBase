import numpy as np

class SoftMaxCrossEntropy(object):

    def __init__(self, layerType):
        self.layerType = layerType
        self.previousLayer = None
        self.forwardOutput = None
        self.backwardOutput = None
        self.target = None

    def ForwardOutput(self):
        self.forwardOutput = np.sum(self.previousLayer.forwardOutput - self.target)

    def BackwardOutput(self):
        self.backwardOutput =  self.previousLayer.forwardOutput - self.target
