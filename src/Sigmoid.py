import math

class Sigmoid(object):

    @classmethod
    def ForwardOutput(self, sigmoidInput):
        sigmoidInput = -sigmoidInput
        sigmoidInput[sigmoidInput > 10.0] = 10.0
        return 1.0/(1.0 + math.e**(sigmoidInput))

    @classmethod
    def BackwardOutput(self, sigmoidInput):
        return sigmoidInput * (1 - sigmoidInput)
