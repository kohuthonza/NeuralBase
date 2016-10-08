import numpy as np

class FullyConnectedLayer:

    __init__(self, numberOfNeurons, activationFunction):
        self.numberOfNeurons = numberOfNeurons
        self.forwardOutput = None
        self.backwardoutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.weights = None

    def ConectLayer(previosLayer, followingLayer):
        self.previousLayer = previosLayer
        self.followingLayer = followingLayer

    def InitializeWeights():
        self.weights = np.zeros((previousLayer.numberOfNeurons, previousLayer.batchSize))
        self.weights = 0.01

    def ForwardOutput():
        self.forwardOutput = np.dot(previousLayer.forwardOutput, self.weights)
