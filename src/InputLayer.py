import numpy as np

class InputLayer(object):

    def __init__(self, dataShape):
        self.numberOfNeurons = dataShape.channels * dataShape.height * dataShape.width
        self.dataShape = dataShape
        self.forwardOutput = np.zeros((1, dataShape.channels, dataShape.height, dataShape.width))

    def ForwardOutput(self, inputData):
        self.forwardOutput = np.zeros((len(inputData), self.dataShape.channels, self.dataShape.height, self.dataShape.width))
        for i, data in enumerate(inputData):
            if self.dataShape.channels == 3:
                data = np.rollaxis(data, 2, 0)
            self.forwardOutput[i] = data
