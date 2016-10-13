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
        self.grayscale = None
        self.inputLayer = None
        self.lossLayer = None
        self.layers = []

    def CreateDataShape(self, inputSample):
        if len(inputSample.shape) == 1:
            self.dataShape.channels = 1
            self.dataShape.height = inputSample.shape[0]
            self.dataShape.width = 1
        elif len(inputSample.shape) == 2:
            self.dataShape.channels = 1
            self.dataShape.height = inputSample.shape[0]
            self.dataShape.width = inputSample.shape[1]
        else:
            self.dataShape.channels = 3
            self.dataShape.height = inputSample.shape[0]
            self.dataShape.width = inputSample.shape[1]

    def CreateLayers(self, layersProperties, lossLayer, inputSample):
        self.inputLayer = InputLayer.InputLayer(inputSample)
        if lossLayer == 'EuclideanDistance':
            self.lossLayer = EuclideanDistance.EuclideanDistance('EuclideanDistance')
        elif lossLayer == 'SoftMaxCrossEntropy':
            self.lossLayer = SoftMaxCrossEntropy.SoftMaxCrossEntropy('SoftMaxCrossEntropy')
        for layerProperties in layersProperties:
            if layerProperties['type'] == 'FullyConnected':
                self.layers.append(FullyConnectedLayer.FullyConnectedLayer(
                                                layerProperties['type'],
                                                layerProperties['numberOfNeurons'],
                                                layerProperties['activationFunction'],
                                                layerProperties['bias']))
    def ConectLayers(self):
        self.inputLayer.followingLayer = self.layers[0]
        self.layers[0].previousLayer = self.inputLayer
        if len(self.layers) > 1:
            self.layers[0].followingLayer = self.layers[1]
            self.layers[-1].previousLayer = self.layers[-2]
        if len(self.layers) > 2:
            for layerIndex in range(1, len(self.layers) - 1):
                self.layers[layerIndex].previousLayer = self.layers[layerIndex - 1]
                self.layers[layerIndex].followingLayer = self.layers[layerIndex + 1]
        self.lossLayer.previousLayer = self.layers[-1]
        self.layers[-1].followingLayer = self.lossLayer

    def InitializeWeights(self):
        for layer in self.layers:
            layer.InitializeWeights()

    def ForwardPropagation(self, input):
        self.inputLayer.ForwardOutput(input, self.dataShape)
        for layer in self.layers:
            layer.ForwardOutput()

    def BackwardPropagation(self, target):
        self.lossLayer.BackwardOutput(target)
        self.layers[-1].backwardOutput = self.lossLayer.backwardOutput
        for layerIndex in range(0, len(self.layers) - 1):
            self.layers[len(self.layers) - layerIndex - 2].BackwardOutput()

    def ActualizeWeights(self, learningRate):
        for layer in self.layers:
            layer.ActualizeWeights(learningRate)
