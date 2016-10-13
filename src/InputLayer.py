import numpy as np

class InputLayer(object):
    
    def __init__(self, inputSample):
        self.numberOfNeurons = inputSample.size
        self.forwardOutput = None

    def ForwardOutput(self, inputData, dataShape):
        self.forwardOutput = np.zeros((len(inputData), dataShape.channels, dataShape.height, dataShape.width))
        for i, data in enumerate(inputData):
            if dataShape.channels == 3:
                data = np.rollaxis(data, 2, 0)
            self.forwardOutput[i] = data
