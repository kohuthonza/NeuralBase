import numpy as np

class EuclideanDistance(object):

    def __init__(self):
        self.layerType = 'EuclideanDistance'
        self.previousLayer = None
        self.forwardOutput = None
        self.backwardOutput = None
        self.target = None

    def ForwardOutput(self):
        self.forwardOutput = (1.0/(2.0 * self.previousLayer.forwardOutput.shape[0])) * (np.sum(np.square(self.previousLayer.forwardOutput - self.target)))

    def BackwardOutput(self):
        self.backwardOutput = (self.previousLayer.forwardOutput - self.target)/self.previousLayer.forwardOutput.shape[0]
