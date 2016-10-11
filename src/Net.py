import DataShape
import InputLayer
import FullyConnectedLayer
import Sigmoid
import SoftMax
import EuclideanDistance
import SoftMaxCrossEntropy

class Net(object):

    def __init__(self):
        self.dataShape = DataShape.DataShape()
        self.inputLayer = None
        self.lossLayer = None
        self.fullyConnectedLayers = []

    def CreateDataShape(self, inputSample):
        if len(inputSample.shape) == 1:
            self.dataShape.channels = 1
            self.dataShape.height = inputSample[0]
            self.dataShape.width = 1
        elif len(inputSample.shape) == 2:
            self.dataShape.channels = 1
            self.dataShape.height = inputSample[0]
            self.dataShape.width = inputSample[1]
        else:
            self.dataShape.channels = 3
            self.dataShape.height = inputSample.shape[0]
            self.dataShape.width = inputSample.shape[1]

    def CreateLayers(self, layersProperties, lossLayer, inputSample):
        self.inputLayer = InputLayer.InputLayer(inputSample)
        if lossLayer == 'EuclideanDistance':
            self.lossLayer = EuclideanDistance.EuclideanDistance()
        elif lossLayer == 'SoftMaxCrossEntropy':
            self.lossLayer = SoftMaxCrossEntropy.SoftMaxCrossEntropy()
        for layerProperties in layersProperties:
            if layerProperties[0] == 'FullyConnected':
                self.fullyConnectedLayers.append(FullyConnectedLayer.FullyConnectedLayer(layerProperties[1], layerProperties[2], layerProperties[3]))

    def ConectLayers(self):
        self.inputLayer.followingLayer = self.fullyConnectedLayers[0]
        self.fullyConnectedLayers[0].previousLayer = self.inputLayer
        if len(self.fullyConnectedLayers) > 1:
            self.fullyConnectedLayers[0].followingLayer = self.fullyConnectedLayers[1]
            self.fullyConnectedLayers[-1].previousLayer = self.fullyConnectedLayers[-2]
        if len(self.fullyConnectedLayers) > 2:
            for layerIndex in range(1, len(self.fullyConnectedLayers) - 1):
                self.fullyConnectedLayers[layerIndex].previousLayer = self.fullyConnectedLayers[layerIndex - 1]
                self.fullyConnectedLayers[layerIndex].followingLayer = self.fullyConnectedLayers[layerIndex + 1]
        self.lossLayer.previousLayer = self.fullyConnectedLayers[-1]
        self.fullyConnectedLayers[-1].followingLayer = self.lossLayer

    def InitializeWeights(self):
        for layer in self.fullyConnectedLayers:
            layer.InitializeWeights()

    def ForwardPropagation(self, input):
        self.inputLayer.ForwardOutput(input, self.dataShape)
        for layer in self.fullyConnectedLayers:
            layer.ForwardOutput()

    def BackwardPropagation(self, target):
        self.lossLayer.BackwardOutput(target)
        self.fullyConnectedLayers[-1].backwardOutput = self.lossLayer.backwardOutput
        for layerIndex in range(0, len(self.fullyConnectedLayers) - 1):
            self.fullyConnectedLayers[len(self.fullyConnectedLayers) - layerIndex - 2].BackwardOutput()

    def ActualizeWeights(self, gamma):
        for layer in self.fullyConnectedLayers:
            layer.ActualizeWeights(gamma)
