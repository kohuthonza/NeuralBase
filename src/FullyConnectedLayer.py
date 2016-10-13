import numpy as np
import random
import Sigmoid
import SoftMax

class FullyConnectedLayer(object):

    def __init__(self, layerType, numberOfNeurons, activationFunction, bias):
        self.layerType = layerType
        self.numberOfNeurons = numberOfNeurons
        self.activationFunction = activationFunction
        self.bias = bias
        self.forwardOutput = None
        self.backwardOutput = None
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
            self.forwardOutput = np.dot(self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1), self.weights)
        else:
            self.forwardOutput = np.dot(self.previousLayer.forwardOutput, self.weights)
        if self.bias is not None:
            self.forwardOutput += self.bias
        if self.activationFunction == 'Sigmoid':
            self.forwardOutput = Sigmoid.Sigmoid.ForwardOutput(self.forwardOutput)
        elif self.activationFunction == 'SoftMax':
            self.forwardOutput = SoftMax.SoftMax.ForwardOutput(self.forwardOutput)

    def BackwardOutput(self):
        self.backwardOutput = np.dot(self.followingLayer.backwardOutput, self.followingLayer.weights.transpose())
        if self.activationFunction == 'Sigmoid':
            self.backwardOutput = self.backwardOutput * Sigmoid.Sigmoid.BackwardOutput(self.forwardOutput)

    def ActualizeWeights(self, learningRate):
        for i, backwardColumn in enumerate(self.backwardOutput.transpose()):
            if (len(self.previousLayer.forwardOutput.shape) == 4):
                self.weights[:,i] -= learningRate * np.sum(backwardColumn.reshape(-1,1) * self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1), axis=0)
            else:
                self.weights[:,i] -= learningRate * np.sum(backwardColumn.reshape(-1,1) * self.previousLayer.forwardOutput, axis=0)
        if self.bias is not None:
            self.bias -= learningRate * np.sum(self.backwardOutput, axis=0)
