import numpy as np
import random

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
        variance = 2/(previousLayer.numberOfNeurons + followingLayer.numberOfNeurons)
        self.weights = np.random.uniform(-variance, variance, (previousLayer.numberOfNeurons, previousLayer.batchSize))

    def ForwardOutput():
        self.forwardOutput = np.dot(previousLayer.forwardOutput, self.weights)
        self.forwardOutput =
