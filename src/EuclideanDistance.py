import numpy as np

class EuclideanDistance(object):

    def __init__(self):
        self.previousLayer = None
        self.lossOutput = None
        self.backwardOutput = None
    def LossOutput(self, target):
        self.lossOutput = (1.0/(2.0 * self.previousLayer.forwardOutput.shape[0])) * (np.sum(np.square(self.previousLayer.forwardOutput - target)))

    def BackwardOutput(self, target):
        self.backwardOutput =  (self.previousLayer.forwardOutput - target)/self.previousLayer.forwardOutput.shape[0]
