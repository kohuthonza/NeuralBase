import numpy as np

class InputLayer:

    def __init__(self, numberOfInputs):
        self.numberOfNeurons = numberOfInputs
        self.batchSize = None
        self.forwardOutput = None

    def ForwardOutput(input):
        self.batchSize = len(input)
        self.forwardOutput = np.zeros((self.batchSize, self.numberOfNeurons))
        for i, data in enumerate(input):
            self.forwardOutput[i] = data.reshape(self.numberOfNeurons)
