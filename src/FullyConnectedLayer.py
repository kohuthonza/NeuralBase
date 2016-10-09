import numpy as np
import random
import Sigmoid

class FullyConnectedLayer(object):

    def __init__(self, numberOfNeurons, activationFunction, bias):
        self.numberOfNeurons = numberOfNeurons
        self.activationFunction = activationFunction
        self.bias = bias
        self.forwardOutput = None
        self.backwardoutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.weights = None

    def InitializeWeights(self):
        variance = 2.0/(self.previousLayer.numberOfNeurons + self.numberOfNeurons)
        self.weights = np.random.uniform(-variance, variance, (self.previousLayer.numberOfNeurons, self.numberOfNeurons))
        if self.bias is not None:
            self.bias = np.full(self.numberOfNeurons, self.bias, dtype=np.float)

    def ForwardOutput(self):
        if (len(self.previousLayer.forwardOutput.shape) == 4):
            if not (self.previousLayer.forwardOutput.shape[1] == 1 and self.previousLayer.forwardOutput.shape[3] == 1):
                self.forwardOutput = np.dot(self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], 1, -1, 1)[:, 0, :, 0], self.weights)
        else:
            self.forwardOutput = np.dot(self.previousLayer.forwardOutput, self.weights)

        if self.bias is not None:
            self.forwardOutput += self.bias

        if self.activationFunction == 'Sigmoid':
            self.forwardOutput = Sigmoid.Sigmoid.ForwardOutput(self.forwardOutput)
        
