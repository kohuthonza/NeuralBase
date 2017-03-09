import math
import numpy as np

class SoftMaxLayer(object):

    def __init__(self):
        self.forwardOutput = None
        self.backwardOutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.numberOfNeurons = None

    def InitializeWeights(self):
        self.numberOfNeurons = self.previousLayer.numberOfNeurons

    def ForwardOutput(self):
        self.forwardOutput = math.e**self.previousLayer.forwardOutput/np.sum(math.e**self.previousLayer.forwardOutput, axis=1).reshape(-1,1)

    def BackwardOutput(self):
        self.backwardOutput = self.followingLayer.backwardOutput

    def ActualizeWeights(self, learningRate):
        pass
