import numpy as np

class BatchNormalizationLayer(object):

    def __init__(self, eps):
        self.eps = eps
        self.variances = None
        self.forwardOutput = None
        self.backwardOutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.numberOfNeurons = None

    def InitializeWeights(self):
        self.numberOfNeurons = self.previousLayer.numberOfNeurons

    def ForwardOutput(self):
        if (len(self.previousLayer.forwardOutput.shape) != 2):
            self.forwardOutput = np.empty_like(self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1))
            self.forwardOutput[:] = self.previousLayer.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], -1)
        else:
            self.forwardOutput = np.empty_like(self.previousLayer.forwardOutput)
            self.forwardOutput[:] = self.previousLayer.forwardOutput

        self.variances = np.zeros(self.forwardOutput.shape[1])
        for i in range(0, self.forwardOutput.shape[1]):
            mean = self.forwardOutput[:,i].mean()
            variance = np.power(self.forwardOutput[:,i] - mean, 2).mean()
            self.forwardOutput[:,i] -= mean
            self.forwardOutput[:,i] /= np.sqrt(variance + self.eps)
            self.variances[i] = variance

    def BackwardOutput(self):
        self.backwardOutput = np.empty_like(self.followingLayer.backwardOutput)
        self.backwardOutput[:] = self.followingLayer.backwardOutput
        for i in range(0, self.backwardOutput.shape[1]):
            self.backwardOutput[:,i] /= np.sqrt(self.variances[i] + self.eps)

    def ActualizeWeights(self, learningRate):
        pass
