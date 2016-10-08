

class Net:

    __init__(self, numberOfInputs, layersProperties):
        
		self.inputLayer = InputLayer(numberOfInputs)
		self.fullyConnectedLayers = []
		
		for layerProperties in layersProperties:
            self.fullyConnectedLayers.add(FullyConnectedLayer(layersProperties[0], layersProperties[1]))

        self.inputLayer.followingLayer = self.fullyConnectedLayers[0]
        self.fullyConnectedLayers[0].previousLayer = self.inputLayer
		
        if len(self.fullyConnectedLayers) > 1:
            self.fullyConnectedLayers[0].followingLayer = self.fullyConnectedLayers[1]
            self.fullyConnectedLayers[1].previosLayer = self.fullyConnectedLayers[0]
			
        if len(self.fullyConnectedLayers) > 2:
            for layerIndex in range(2, len(self.fullyConnectedLayers - 1)):
                self.fullyConnectedLayers[layerIndex].previousLayer = self.fullyConnectedLayers[layerIndex - 1]
                self.fullyConnectedLayers[layerIndex].followingLayer = self.fullyConnectedLayers[layerIndex + 1]
		
		for layer in fullyConnectedLayers:
			layer.InitializeWeights()
		

    def forwardPropagation(input):
        self.inputLayer.ForwardOutput(input)
        for layer in self.fullyConnectedLayers:
            layer.ForwardOutput()
