import numpy as np

class ConvolutionalLayer(object):
    def __init__(self, kernelSize, numberOfKernels, stride):
        self.kernelSize = kernelSize
        self.numberOfKernels = numberOfKernels
        self.stride = stride
        self.numberOfNeurons = None
        self.mappingMatrix = None
        self.forwardOutput = None
        self.backwardOutput = None
        self.followingLayer = None
        self.previousLayer = None
        self.kernelsWeights = None

    def InitializeWeights(self):
        inputDataSize =  self.previousLayer.forwardOutput.shape[2] * self.previousLayer.forwardOutput.shape[3]
        kernelGap = self.kernelSize/2
        numberOfKernelPasses = inputDataSize - (4 * (kernelGap) ** 2)
        numberOfWeightsInKernel = self.previousLayer.forwardOutput.shape[1] * self.kernelSize * self.kernelSize
        self.mappingMatrix = np.zeros((numberOfWeightsInKernel, numberOfKernelPasses))

        mappingColumnIndex = 0
        mappingRowIndex = 0
        for rowIndex in self.previousLayer.forwardOutput.shape[3]/self.stride:
            for columnIndex in self.previousLayer.forwardOutput.shape[2]/self.stride:
                for kernelColumnIndex in self.kernelSize:
                    for kernelRowIndex in self.kernelSize:
                        for channelIndex in self.previousLayer.forwardOutput.shape[1]:
                            self.mappingMatrix[:, mappingColumnIndex][mappingRowIndex] = (rowIndex * self.stride + kernelRowIndex + kernelGap,
                                                                   columnIndex * self.stride + kernelColumnIndex + kernelGap,
                                                                   channelIndex)
                            mappingRowIndex += 1
                mappingRowIndex = 0
                mappingColumnIndex += 1

        self.numberOfNeurons = self.numberOfKernels * self.previousLayer.numberOfNeurons
        self.kernelsWeights = np.zeros(self.numberOfKernels,
                                       self.previousLayer.forwardOutput[1],
                                       self.kernelSize,
                                       self.kernelSize)
        variance = 2.0/(self.previousLayer.numberOfNeurons + self.numberOfNeurons)
        for kernelIndex in self.numberOfKernels:
            self.kernelsWeights[kernelIndex] = np.random.uniform(variance, 0,
                                                                (self.previousLayer.forwardOutput[1],
                                                                 self.kernelSize,
                                                                 self.kernelSize))

    def ForwardOutput(self):

        weights = np.zeros(self.numberOfKernels, self.kernelSize * self.kernelSize * self.previousLayer.forwardOutput[1])

        self.forwardOutput = np.zeros(self.previousLayer.forwardOutput.shape[0], self.mappingMatrix.shape[0], self.mappingMatrix.shape[1])
        for batchIndex in self.previousLayer.forwardOutput.shape[0]:
            for kernelColumnIndex in self.kernelSize:
                for kernelRowIndex in self.kernelSize:
                    for channelIndex in self.previousLayer.forwardOutput.shape[1]:
                        weights[batchIndex][(kernelColumnIndex + kernelRowIndex)*self.previousLayer.forwardOutput.shape[1] + channelIndex] = kernelsWeights[batchIndex][channelIndex][rowIndex][columnIndex]

            for columnIndex in self.mappingMatrix.shape[1]:
                for index in self.mappingMatrix[:, columnIndex]:
                    self.forwardOutput[batchIndex][:, columnIndex] = self.previousLayer.forwardOutput[batchIndex][index[0]][index[1]][index[2]]

        for batchIndex in self.previousLayer.forwardOutput.shape[0]:
            self.forwardOutput[batchIndex] = np.dot(weights[batchIndex], self.forwardOutput[batchIndex])
            self.forwardOutput[batchIndex] = self.forwardOutput[batchIndex].reshape(self.previousLayer.forwardOutput[2] - self.kernelSize/2, self.previousLayer.forwardOutput[3])


#    def BackwardOutput(self):

#    def ActualizeWeights(self, learningRate):
