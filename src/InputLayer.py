import numpy as np

class InputLayer(object):

    def __init__(self, inputSample):
        self.numberOfNeurons = inputSample.size
        self.forwardOutput = None

    def ForwardOutput(self, input, dataShape):
        self.forwardOutput = np.zeros((len(input), dataShape.channels, dataShape.height, dataShape.width))
        for i, data in enumerate(input):
            if dataShape.channels == 3:
                data = np.rollaxis(data, 2, 0)
            self.forwardOutput[i] = data
