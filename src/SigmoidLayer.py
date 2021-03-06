import math
import numpy as np

class SigmoidLayer(object):

    def __init__(self):
        self.forwardOutput = None
        self.backwardOutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.numberOfNeurons = None

    def InitializeWeights(self):
        self.numberOfNeurons = self.previousLayer.numberOfNeurons

    def ForwardOutput(self):
        self.forwardOutput = np.empty_like(self.previousLayer.forwardOutput)
        self.forwardOutput[:] = -self.previousLayer.forwardOutput
        self.forwardOutput[self.forwardOutput > 10.0] = 10.0
        self.forwardOutput = 1.0/(1.0 + math.e**(self.forwardOutput))

    def BackwardOutput(self):
        self.backwardOutput = np.empty_like(self.followingLayer.backwardOutput)
        self.backwardOutput[:] = self.followingLayer.backwardOutput * (self.forwardOutput * (1 - self.forwardOutput))

    def ActualizeWeights(self, learningRate):
        pass
