import numpy as np

class InputLayer(object):

    def __init__(self, dataShape):
        self.numberOfNeurons = dataShape.channels * dataShape.height * dataShape.width
        self.forwardOutput = None

    def ForwardOutput(self, input, dataShape):
        self.forwardOutput = np.zeros((dataShape.batchSize, dataShape.channels, dataShape.height, dataShape.width))
        for i, data in enumerate(input):
            if dataShape.channels == 3:
                data = np.rollaxis(data, 2, 0)
            self.forwardOutput[i] = data
