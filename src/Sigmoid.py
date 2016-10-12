import math

class Sigmoid(object):
    @classmethod
    def ForwardOutput(self, input):
        input = -input
        input[input > 10.0] = 10.0
        return 1.0/(1.0 + math.e**(input))

    @classmethod
    def BackwardOutput(self, input):
        return input * (1 - input)
