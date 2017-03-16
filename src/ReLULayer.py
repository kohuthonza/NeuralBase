import math
import numpy as np

class ReLULayer(object):

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
        self.forwardOutput[:] = self.previousLayer.forwardOutput
        self.forwardOutput[self.forwardOutput < 0] = 0

    def BackwardOutput(self):
        self.backwardOutput = np.empty_like(self.followingLayer.backwardOutput)
        self.backwardOutput[:] = self.followingLayer.backwardOutput
        self.backwardOutput[self.previousLayer.forwardOutput <= 0] = 0


    def ActualizeWeights(self, learningRate):
        pass
