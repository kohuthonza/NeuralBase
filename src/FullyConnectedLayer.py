import numpy as np
import random

class FullyConnectedLayer(object):

    def __init__(self, numberOfNeurons, bias):
        self.numberOfNeurons = numberOfNeurons
        self.bias = bias
        self.forwardOutput = None
        self.backwardOutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.weights = None

    def InitializeWeights(self):
        variance = np.sqrt(3.0 / (self.previousLayer.numberOfNeurons + self.numberOfNeurons))
        self.weights = np.random.uniform(-variance, variance, (self.previousLayer.numberOfNeurons, self.numberOfNeurons))
        if self.bias is not None:
            self.bias = np.full(self.numberOfNeurons, self.bias, dtype=np.float32)

    def ForwardOutput(self):
        if (len(self.previousLayer.forwardOutput.shape) != 2):
            self.forwardOutput = np.dot(self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1), self.weights)
        else:
            self.forwardOutput = np.dot(self.previousLayer.forwardOutput, self.weights)
        if self.bias is not None:
            self.forwardOutput += self.bias

    def BackwardOutput(self):
        self.backwardOutput = np.dot(self.followingLayer.backwardOutput, self.weights.transpose())

    def ActualizeWeights(self, learningRate):
        weightGradients = np.zeros(self.weights.shape)
        for i, backwardColumn in enumerate(self.followingLayer.backwardOutput.transpose()):
            if (len(self.previousLayer.forwardOutput.shape) == 4):
                weightGradients[:,i] = np.sum(backwardColumn.reshape(-1,1) * self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1), axis=0)
            else:
                weightGradients[:,i] = np.sum(backwardColumn.reshape(-1,1) * self.previousLayer.forwardOutput, axis=0)

        self.weights -= learningRate * weightGradients
        if self.bias is not None:
            self.bias -= learningRate * np.sum(self.followingLayer.backwardOutput, axis=0)
