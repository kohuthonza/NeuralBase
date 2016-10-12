import numpy as np

class SoftMaxCrossEntropy(object):

    def __init__(self):
        self.previousLayer = None
        self.lossOutput = None
        self.backwardOutput = None
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__.update(d)

    def LossOutput(self, target):
        self.lossOutput = np.sum(self.previousLayer.forwardOutput - target)

    def BackwardOutput(self, target):
        self.backwardOutput =  self.previousLayer.forwardOutput - target
