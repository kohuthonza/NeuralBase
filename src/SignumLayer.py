import numpy as np

class SignumLayer(object):

    def __init__(self):
        self.forwardOutput = None
        self.backwardOutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.numberOfNeurons = None

    def InitializeWeights(self):
        self.numberOfNeurons = self.previousLayer.numberOfNeurons

    def ForwardOutput(self):
        self.forwardOutput = np.sign(self.previousLayer.forwardOutput)

    def BackwardOutput(self):
        self.backwardOutput = self.followingLayer.backwardOutput
        self.backwardOutput[self.backwardOutput < -1] = 0
        self.backwardOutput[self.backwardOutput > 1] = 0

    def ActualizeWeights(self, learningRate):
        pass
