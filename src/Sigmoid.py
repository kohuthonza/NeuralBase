import math		

class Sigmoid:
	@staticmethod
	def ForwardOutput(input):
		return 1.0/(1.0 + math.e**(-input))
	
	@staticmethod
	def BackwardOutput(input):
		return input * (1 - input)
