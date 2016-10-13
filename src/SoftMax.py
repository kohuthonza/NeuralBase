import math
import numpy as np

class SoftMax(object):
    
    @classmethod
    def ForwardOutput(self, softMaxInput):
        return math.e**softMaxInput/np.sum(math.e**softMaxInput, axis=1).reshape(-1,1)
