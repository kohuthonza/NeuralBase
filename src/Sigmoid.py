import math

class Sigmoid(object):
    @classmethod
    def ForwardOutput(self, input):
        return 1.0/(1.0 + math.e**(-input))

    @classmethod
    def BackwardOutput(self, input):
        return input * (1 - input)
