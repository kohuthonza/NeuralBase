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
        numberOfKernelPasses = inputDataSize - self.previousLayer.forwardOutput.shape[2] * kernelGap * 2 \
                                             - self.previousLayer.forwardOutput.shape[3] * kernelGap * 2 \
                                             + 4 * (kernelGap**2)
        numberOfWeightsInKernel = self.previousLayer.forwardOutput.shape[1] * self.kernelSize * self.kernelSize
        self.mappingMatrix = np.zeros((numberOfWeightsInKernel, numberOfKernelPasses, 3))

        mappingColumnIndex = 0
        mappingRowIndex = 0
        for rowIndex in range(0, self.previousLayer.forwardOutput.shape[3]/self.stride - 2 * kernelGap):
            for columnIndex in range(0, self.previousLayer.forwardOutput.shape[2]/self.stride - 2 * kernelGap):
                for kernelColumnIndex in range(0, self.kernelSize):
                    for kernelRowIndex in range(0, self.kernelSize):
                        for channelIndex in range(0, self.previousLayer.forwardOutput.shape[1]):
                            self.mappingMatrix[:, mappingColumnIndex][mappingRowIndex] = (channelIndex, rowIndex * self.stride + kernelRowIndex, columnIndex * self.stride + kernelColumnIndex)
                            mappingRowIndex += 1
                mappingRowIndex = 0
                mappingColumnIndex += 1
        self.mappingMatrix = self.mappingMatrix.astype(np.int)
        self.numberOfNeurons = self.numberOfKernels * self.previousLayer.numberOfNeurons
        self.kernelsWeights = np.zeros((self.numberOfKernels,
                                        self.previousLayer.forwardOutput.shape[1],
                                        self.kernelSize,
                                        self.kernelSize))
        variance = 2.0/(self.previousLayer.numberOfNeurons + self.numberOfNeurons)
        for kernelIndex in range(0, self.numberOfKernels):
            self.kernelsWeights[kernelIndex] = np.random.uniform(variance, 0,
                                                                (self.previousLayer.forwardOutput.shape[1],
                                                                 self.kernelSize,
                                                                 self.kernelSize))
        self.forwardOutput = np.zeros((1, self.numberOfKernels, self.previousLayer.forwardOutput.shape[2] - 2 * kernelGap, self.previousLayer.forwardOutput.shape[3] - 2 * kernelGap))
        self.numberOfNeurons = self.numberOfKernels * (self.previousLayer.forwardOutput.shape[2] - 2 * kernelGap) * (self.previousLayer.forwardOutput.shape[3] - 2 * kernelGap)


    def ForwardOutput(self):

        weights = np.zeros((self.numberOfKernels, self.kernelSize * self.kernelSize * self.previousLayer.forwardOutput.shape[1]))
        for kernelIndex in range(0, self.numberOfKernels):
            for kernelColumnIndex in range(0, self.kernelSize):
                for kernelRowIndex in range(0, self.kernelSize):
                    for channelIndex in range(0, self.previousLayer.forwardOutput.shape[1]):
                        weights[kernelIndex][(kernelColumnIndex + kernelRowIndex) * self.previousLayer.forwardOutput.shape[1] + channelIndex] = self.kernelsWeights[kernelIndex][channelIndex][kernelRowIndex][kernelColumnIndex]


        tmpForwardOutput = np.zeros((self.previousLayer.forwardOutput.shape[0], self.mappingMatrix.shape[0], self.mappingMatrix.shape[1]))
        for batchIndex in range(0, self.previousLayer.forwardOutput.shape[0]):
            for columnIndex in range(0, self.mappingMatrix.shape[1]):
                for i, index in enumerate(self.mappingMatrix[:, columnIndex]):
                    tmpForwardOutput[batchIndex][:, columnIndex][i] = self.previousLayer.forwardOutput[batchIndex][index[0]][index[1]][index[2]]

        self.forwardOutput = np.zeros((self.previousLayer.forwardOutput.shape[0],
                                      self.numberOfKernels,
                                      (self.previousLayer.forwardOutput.shape[2] - 2 * (self.kernelSize/2)) * (self.previousLayer.forwardOutput.shape[3] - 2 * (self.kernelSize/2))))
        for batchIndex in range(0, self.previousLayer.forwardOutput.shape[0]):
            for kernelIndex in range(0, self.numberOfKernels):
                self.forwardOutput[batchIndex][kernelIndex] = np.dot(weights[kernelIndex], tmpForwardOutput[batchIndex])

        self.forwardOutput = self.forwardOutput.reshape(self.previousLayer.forwardOutput.shape[0], self.numberOfKernels, self.previousLayer.forwardOutput.shape[2] - 2 * (self.kernelSize/2), self.previousLayer.forwardOutput.shape[3] - 2 * (self.kernelSize/2))

#    def BackwardOutput(self):

#    def ActualizeWeights(self, learningRate):
