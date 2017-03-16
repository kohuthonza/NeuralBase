import numpy as np
import random

class BinaryFullyConnectedLayer(object):

    def __init__(self, numberOfNeurons, bias):
        self.numberOfNeurons = numberOfNeurons
        self.bias = bias
        self.forwardOutput = None
        self.backwardOutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.weights = None
        self.binaryWeights = None
        self.alpha = None

    def InitializeWeights(self):
        variance = np.sqrt(3.0 / (self.previousLayer.numberOfNeurons + self.numberOfNeurons))
        self.weights = np.random.uniform(-variance, variance, (self.previousLayer.numberOfNeurons, self.numberOfNeurons))
        if self.bias is not None:
            self.bias = np.full(self.numberOfNeurons, self.bias, dtype=np.float32)

    def ForwardOutput(self):
        self.alpha = np.abs(self.weights).mean()
        self.binaryWeights = np.sign(self.weights)
        self.binaryWeights *= self.alpha
        if (len(self.previousLayer.forwardOutput.shape) != 2):
            self.forwardOutput = np.dot(self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1), self.binaryWeights)
        else:
            self.forwardOutput = np.dot(self.previousLayer.forwardOutput, self.binaryWeights)
        if self.bias is not None:
            self.forwardOutput += self.bias

    def BackwardOutput(self):
        self.backwardOutput = np.dot(self.followingLayer.backwardOutput, self.binaryWeights.transpose())

    def ActualizeWeights(self, learningRate):

        weightGradients = np.zeros(self.weights.shape)
        for i, backwardColumn in enumerate(self.followingLayer.backwardOutput.transpose()):
            if (len(self.previousLayer.forwardOutput.shape) == 4):
                weightGradients[:,i] = np.sum(backwardColumn.reshape(-1,1) * self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1), axis=0)
            else:
                weightGradients[:,i] = np.sum(backwardColumn.reshape(-1,1) * self.previousLayer.forwardOutput, axis=0)

        """
        gradientUpdate = np.zeros(self.weights.shape)
        gradientUpdate.fill(self.alpha)
        gradientUpdate[self.weights > 1] = 0
        gradientUpdate[self.weights < -1] = 0
        gradientUpdate += 1.0/self.weights.size

        weightGradients *= gradientUpdate
        """

        self.weights -= learningRate * weightGradients
        if self.bias is not None:
            self.bias -= learningRate * np.sum(self.followingLayer.backwardOutput, axis=0)
