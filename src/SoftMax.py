import math
import numpy as np

class SoftMax(object):
    @classmethod
    def ForwardOutput(self, input):
        return math.e**input/np.sum(math.e**input, axis=1).reshape(-1,1)
