import cv2
import sys
import os
import argparse
import json
import sys
import numpy as np
import Net

def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="NeuralBase")
    parser.add_argument('-n', '--net',
                        type=str,
                        required=True,
                        help='JSON specification of net')
    parser.add_argument('-t', '--train',
                        type=str,
                        required=True,
                        help='JSON specification of train process')
    args = parser.parse_args()

    return args

def parseJSONNetSpecification(jsonNetSpecification):

    netSpecification = {}

    layersProperties = ['type', 'numberOfNeurons', 'bias', 'activationFunction']
    layersSpecifications = []

    if 'inputSample' in jsonNetSpecification:
        netSpecification['inputSample'] = jsonNetSpecification['inputSample']
    else:
        sys.exit("JSON file does't specify path to input sample.")
    if 'grayscale' in jsonNetSpecification:
        if netSpecification[trainProperty] == 'True' or trainSpecification[trainProperty] == 'true':
            trainSpecification[trainProperty] = 0
        else:
            trainSpecification[trainProperty] = 1
    else:
        netSpecification['grayscale'] = 1

    if 'layers' in jsonNetSpecification:
        for jsonLayerSpecification in jsonNetSpecification['layers']:
            layer = {}
            for layerProperty in layersProperties:
                if layerProperty in jsonLayerSpecification:
                    layer[layerProperty] = jsonLayerSpecification[layerProperty]
                    if layerProperty == 'numberOfNeurons':
                        layer[layerProperty] = int(layer[layerProperty])
                    elif layerProperty == 'bias':
                        layer[layer] = float(layer[layerProperty])
                else:
                    if layerProperty == 'type':
                        sys.exit("JSON net file doesn't specify type of some layer.")
                    elif layerProperty == 'numberOfNeurons':
                        sys.exit("JSON net file doesn't specify number of neurons in some layer.")
                    layer[layerProperty] = None
            layersSpecifications.append(layer)
    else:
        sys.exit("JSON net file doesn't specify layers.")

    netSpecification['layers'] = layersSpecifications

    if 'lossFunction' in jsonNetSpecification:
        netSpecification['lossFunction'] = jsonNetSpecification['lossFunction']
    else:
        sys.exit("JSON net file doesn't specify loss layer.")


    return netSpecification

def parseJSONTrainSpecification(jsonTrainSpecification):

    trainSpecificationProperties = ['trainData', 'trainLabels',
                                    'testData', 'testLabels'
                                    'batchSize',
                                    'meanData', 'scaleData',
                                    'meanLabels', 'scaleLabels',
                                    'grayscale',
                                    'learningRate',
                                    'learnignRateDrop', 'dropFrequency',
                                    'numberOfTrainIterations' 'numberOfTestIterations', 'testFrequency']

    trainSpecification = {}

    for trainProperty in trainSpecificationProperties:
        if trainProperty in jsonTrainSpecification:
            trainSpecification[trainProperty] = jsonTrainSpecification[trainProperty]
            if trainProperty == 'batchSize' or \
               trainProperty == 'dropFrequency' or \
               trainProperty == 'numberOfTrainIterations' or \
               trainProperty == 'numberOfTestIterations' or \
               trainProperty == 'trainOutputFrequency' or \
               trainProperty == 'testFrequency':
                trainSpecification[trainProperty] = int(trainSpecification[trainProperty])
            elif trainProperty == 'mean' or \
                 trainProperty == 'scale' or \
                 trainProperty == 'learningRate' or \
                 trainProperty == 'learnignRateDrop':
                trainSpecification[trainProperty] = float(trainSpecification[trainProperty])

        else:
            if trainProperty == 'trainData':
                sys.exit("JSON train file doesn't specify folder with training data.")
            elif trainProperty == 'trainLabels':
                sys.exit("JSON train file doesn't specify file with training labels.")
            elif trainProperty == 'batchSize':
                trainSpecification[trainProperty] = 32
            elif trainProperty == 'numberOfTrainIterations':
                sys.exit("JSON train file doesn't specify number of train iteration.")
            elif trainProperty == 'learningRate':
                sys.exit("JSON train file doesn't specify learning rate.")
            trainSpecification[trainProperty] = None

    return trainSpecification

def PrepareLabels(trainSpecification):

    loadLabels = {}

    for label in ['tarinLabels', 'testLabels']:
        with open(jsonTrainSpecification[dataset]) as f:
            loadLabels[datasetLabel] = f.read().splitlines()
        for dataIndex in range(len(jsonTrainSpecification[dataset])):
            loadLabels[datasetLabel][dataIndex] = jsonTrainSpecification[datasetLabel][dataIndex].split()

    return loadLabels

def PrepareInput(data, dataDirectory, batchSize, meanData, scaleData, meanLabels, scaleLabels, grayscale):
    batch = {}
    batch['dataBatch'] = []
    batch['labelsBatch'] = []
    randomIndexes = np.random.randint(len(data), size=batchSize)
    for index in randomIndexes:
        img = cv2.imread(os.path.join(dataDirectory + data[index][0]), grayscale)
        if meanData is not None:
            img -= meanData
        if scaleData is not None:
            img = img/scaleData
        batch['dataBatch'].append(img)
        labels = data[index][1:]
        labels = [float(tmpLabel) for tmpLabel in labels]
        if meanLabels is not None:
            labels -= meanLabels
        if scaleLabels is not None:
            labels = label/scaleLabels
        bacth['labelsBatch'].append(labels)
    batchLabels = np.asarray(batchLabels)

    return batch

def CreateNet(netSpecification):

    net = Net.Net()
    inputSample = cv2.imread(inputSample, netSpecification['grayscale'])
    net.CreateDataShape(inputSample)
    net.CreateLayers(layersSpecifications, jsonNetSpecification['lossFunction'], inputSample)
    net.ConectLayers()
    net.InitializeWeights()

    return net

def TrainNet(trainSpecification, net):

    labels = PrepareLabels(trainSpecification)

    iterTrainLoss = 0
    iterTestLoss = 0
    trainCounter = 0
    outputTrainCounter = 0
    testCounter = 0
    dropFrequencyCounter = 0

    for trainCounter in range(trainSpecification['numberOfTrainIterations']):
        batch = PrepareInput(labels['trainLabels'],
                            trainSpecification['trainData'],
                            trainSpecification['bacthSize'],
                            trainSpecification['meanData'],
                            trainSpecification['scaleData'],
                            trainSpecification['meanLabels'],
                            trainSpecification['scaleLabels'],
                            grayscale)

        net.ForwardPropagation(bacth['dataBatch'])
        net.lossLayer.LossOutput(batch['labelsBatch'])
        net.BackwardPropagation(batch['labelsBatch'])
        net.ActualizeWeights(trainSpecification['learningRate'])

        iterTrainLoss += net.lossLayer.lossOutput

        trainCounter += 1
        testCounter += 1
        outputTrainCounter += 1

        if trainSpecification['dropFrequency'] is not None:
            dropFrequencyCounter += 1
            if trainSpecification['dropFrequency'] == dropFrequencyCounter:
                trainSpecification['learningRate'] *= trainSpecification['learnignRateDrop']
                dropFrequencyCounter = 0

        if trainSpecification['trainOutputFrequency'] is not None:
            if trainSpecification['trainOutputFrequency'] == outputTrainCounter:
                outputTrainCounter = 0
                print("")
                print("")
                print('{} train iteration DONE.'.format(counter))
                print("Train loss: {}".format(iterTrainLoss/trainSpecification['trainOutputFrequency']))
                iterTrainLoss = 0

        if trainSpecification['testFrequency'] is not None:
            if testCounter == trainSpecification['testFrequency']:
                testCounter = 0
                accuracy = 0
                print("")
                print("")
                print('********************************************')
                print('Testing net...')
                for counter in range(trainSpecification['numberOfTestIterations']):
                    batch = PrepareInput(labels['testLabels'],
                                        jsonTrainSpecification['testData'],
                                        trainSpecification['bacthSize'],
                                        trainSpecification['meanData'],
                                        trainSpecification['scaleData'],
                                        trainSpecification['meanLabels'],
                                        trainSpecification['scaleLabels'],
                                        grayscale)

                    net.ForwardPropagation(batch['dataBatch'])
                    net.lossLayer.LossOutput(batch['labelsBatch'])
                    iterTestLoss += net.lossLayer.lossOutput

                    for batchIndex in range(trainSpecification['batchSize']):
                        if trainSpecification['lossLayer'] == 'SoftMaxCrossEntropy':
                            if np.argmax(net.fullyConnectedLayers[-1].forwardOutput[batchIndex]) == np.argmax(batchLabels[batchIndex]):
                                accuracy += 1
                print('____________________________________________')
                print('Last batch from training set:')
                print("Net: {}".format(net.fullyConnectedLayers[-1].forwardOutput[batchIndex]))
                print("Label: {}".format(batchLabels[batchIndex]))
                print('{} test iteration DONE.'.format(trainSpecification['numberOfTestIterations']))
                if trainSpecification['lossLayer'] == 'SoftMaxCrossEntropy':
                    print("Accuracy: {}".format(float(accuracy)/(trainSpecification['numberOfTestIterations'] * trainSpecification['bacthSize'])))
                else:
                    print("Test loss: {}".format(iterTestLoss/trainSpecification['numberOfTestIterations']))

                iterTestLoss = 0

    if iterTrainLoss != 0:
        print('{} train iteration DONE.'.format(outputTrainCounter))
        print("Train loss: {}".format(iterTrainLoss/outputTrainCounter))


def main():
    args = parse_args()

    with open(args.net) as jsonFile:
        jsonNetSpecification = json.load(jsonFile)
    with open(args.train) as jsonFile:
        jsonTrainSpecification = json.load(jsonFile)

    print("")
    print("")
    print('********************************************')
    print('Net specification')
    print('********************************************')
    print("")
    print json.dumps(jsonNetSpecification, indent=2, sort_keys=False)
    print("")
    print("")
    print("")
    print('********************************************')
    print('Train specification')
    print('********************************************')
    print("")
    print json.dumps(jsonTrainSpecification, indent=2, sort_keys=False)
    print("")
    print("")
    print("")

    netSpecification = parseJSONNetSpecification(jsonNetSpecification)
    trainSpecification = parseJSONTrainSpecification(jsonTrainSpecification)

    #net = CreateNet(netSpecification)
    #net = TrainNet(trainSpecification, net)

if __name__ == "__main__":
    main()
