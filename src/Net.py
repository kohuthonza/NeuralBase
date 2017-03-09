import DataShape
import InputLayer
import ConvolutionalLayer
import FullyConnectedLayer
import SigmoidLayer
import ReLULayer
import SoftMaxLayer
import BatchNormalizationLayer
import SignumLayer
import BinaryFullyConnectedLayer
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

    def CreateLayers(self, layersProperties, lossLayer):
        self.inputLayer = InputLayer.InputLayer(self.dataShape)
        if lossLayer == 'EuclideanDistance':
            self.lossLayer = EuclideanDistance.EuclideanDistance()
        elif lossLayer == 'SoftMaxCrossEntropy':
            self.lossLayer = SoftMaxCrossEntropy.SoftMaxCrossEntropy()
        for layerProperties in layersProperties:
            if layerProperties['type'] == 'FullyConnected':
                self.layers.append(FullyConnectedLayer.FullyConnectedLayer(
                                                layerProperties['numberOfNeurons'],
                                                layerProperties['bias']))
            elif layerProperties['type'] == 'BinaryFullyConnected':
                self.layers.append(BinaryFullyConnectedLayer.BinaryFullyConnectedLayer(
                                            layerProperties['numberOfNeurons'],
                                            layerProperties['bias']))
            elif layerProperties['type'] == 'BatchNormalization':
                self.layers.append(BatchNormalizationLayer.BatchNormalizationLayer(
                                                layerProperties['eps']))
            elif layerProperties['type'] == 'Signum':
                self.layers.append(SignumLayer.SignumLayer())
            elif layerProperties['type'] == 'Sigmoid':
                self.layers.append(SigmoidLayer.SigmoidLayer())
            elif layerProperties['type'] == 'ReLU':
                self.layers.append(ReLULayer.ReLULayer())
            elif layerProperties['type'] == 'SoftMax':
                self.layers.append(SoftMaxLayer.SoftMaxLayer())
            elif layerProperties['type'] == 'Convolutional':
                self.layers.append(ConvolutionalLayer.ConvolutionalLayer(
                                                layerProperties['kernelSize'],
                                                layerProperties['numberOfKernels'],
                                                layerProperties['stride']))
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

    def ForwardPropagation(self, dataInput):
        self.inputLayer.ForwardOutput(dataInput)
        for layer in self.layers:
            layer.ForwardOutput()

    def BackwardPropagation(self, target):
        self.lossLayer.target = target
        self.lossLayer.BackwardOutput()
        for layerIndex in range(0, len(self.layers) - 1):
            self.layers[len(self.layers) - layerIndex - 1].BackwardOutput()

    def ActualizeWeights(self, learningRate):
        for layer in self.layers:
            layer.ActualizeWeights(learningRate)
