import cv2
import sys
import os
import argparse
import json
import sys
from collections import OrderedDict
import numpy as np
import pickle
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
    parser.add_argument('-o', '--output',
                        type=str,
                        help='Name of net to be saved')
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
        if netSpecification['grayscale'] == 'True' or trainSpecification['grayscale'] == 'true':
            trainSpecification['grayscale'] = 0
        else:
            trainSpecification['grayscale'] = 1
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
                        layer[layerProperty] = float(layer[layerProperty])
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
                                    'testData', 'testLabels',
                                    'batchSize',
                                    'meanData', 'scaleData',
                                    'meanLabels', 'scaleLabels',
                                    'grayscale',
                                    'learningRate',
                                    'learnignRateDrop', 'dropFrequency',
                                    'numberOfTrainIterations', 'trainOutputFrequency',
                                    'numberOfTestIterations', 'testOutputFrequency']

    trainSpecification = {}

    for trainProperty in trainSpecificationProperties:
        if trainProperty in jsonTrainSpecification:
            trainSpecification[trainProperty] = jsonTrainSpecification[trainProperty]
            if trainProperty == 'batchSize' or \
               trainProperty == 'dropFrequency' or \
               trainProperty == 'numberOfTrainIterations' or \
               trainProperty == 'numberOfTestIterations' or \
               trainProperty == 'trainOutputFrequency' or \
               trainProperty == 'testOutputFrequency':
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

    for datasetLabel in ['trainLabels', 'testLabels']:
        with open(trainSpecification[datasetLabel]) as f:
            loadLabels[datasetLabel] = f.read().splitlines()
        for dataIndex in range(len(loadLabels[datasetLabel])):
            loadLabels[datasetLabel][dataIndex] = loadLabels[datasetLabel][dataIndex].split()

    return loadLabels

def PrepareInput(dataLabels, dataDirectory, batchSize, meanData, scaleData, meanLabels, scaleLabels, grayscale):
    batch = {}
    batch['dataBatch'] = []
    batch['labelsBatch'] = []
    randomIndexes = np.random.randint(len(dataLabels), size=batchSize)
    for index in randomIndexes:
        img = cv2.imread(os.path.join(dataDirectory, dataLabels[index][0]), grayscale)
        if meanData is not None:
            img -= meanData
        if scaleData is not None:
            img = img/scaleData
        batch['dataBatch'].append(img)
        labels = dataLabels[index][1:]
        labels = [float(tmpLabel) for tmpLabel in labels]
        if meanLabels is not None:
            labels -= meanLabels
        if scaleLabels is not None:
            labels = label/scaleLabels
        batch['labelsBatch'].append(labels)
    batch['labelsBatch'] = np.asarray(batch['labelsBatch'])

    return batch

def CreateNet(netSpecification):

    net = Net.Net()
    inputSample = cv2.imread(netSpecification['inputSample'], netSpecification['grayscale'])
    net.CreateDataShape(inputSample)
    net.CreateLayers(netSpecification['layers'], netSpecification['lossFunction'], inputSample)
    net.ConectLayers()
    net.InitializeWeights()

    return net

def TrainNet(trainSpecification, netSpecification, net):

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
                            trainSpecification['batchSize'],
                            trainSpecification['meanData'],
                            trainSpecification['scaleData'],
                            trainSpecification['meanLabels'],
                            trainSpecification['scaleLabels'],
                            netSpecification['grayscale'])

        net.ForwardPropagation(batch['dataBatch'])
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
                print("____________________________________________")
                print('{} train iteration DONE.'.format(trainCounter))
                print("Train loss: {}".format(iterTrainLoss/trainSpecification['trainOutputFrequency']))
                iterTrainLoss = 0

        if trainSpecification['testOutputFrequency'] is not None:
            if testCounter == trainSpecification['testOutputFrequency']:
                testCounter = 0
                accuracy = 0
                print("")
                print("")
                print('********************************************')
                print('Testing net...')
                for counter in range(trainSpecification['numberOfTestIterations']):
                    batch = PrepareInput(labels['testLabels'],
                                        trainSpecification['testData'],
                                        trainSpecification['batchSize'],
                                        trainSpecification['meanData'],
                                        trainSpecification['scaleData'],
                                        trainSpecification['meanLabels'],
                                        trainSpecification['scaleLabels'],
                                        netSpecification['grayscale'])

                    net.ForwardPropagation(batch['dataBatch'])
                    net.lossLayer.LossOutput(batch['labelsBatch'])
                    iterTestLoss += net.lossLayer.lossOutput

                    for batchIndex in range(trainSpecification['batchSize']):
                        if netSpecification['lossFunction'] == 'SoftMaxCrossEntropy':
                            if np.argmax(net.fullyConnectedLayers[-1].forwardOutput[batchIndex]) == np.argmax(batch['labelsBatch'][batchIndex]):
                                accuracy += 1
                print('____________________________________________')
                print('Last batch from training set:')
                print("Net: {}".format(net.fullyConnectedLayers[-1].forwardOutput[batchIndex]))
                print("Label: {}".format(batch['labelsBatch'][batchIndex]))
                print('{} test iteration DONE.'.format(trainSpecification['numberOfTestIterations']))
                if netSpecification['lossFunction'] == 'SoftMaxCrossEntropy':
                    print("Accuracy: {}".format(float(accuracy)/(trainSpecification['numberOfTestIterations'] * trainSpecification['batchSize'])))
                else:
                    print("Test loss: {}".format(iterTestLoss/trainSpecification['numberOfTestIterations']))
                print('********************************************')
                print("")
                iterTestLoss = 0

    if iterTrainLoss != 0:
        print('{} train iteration DONE.'.format(outputTrainCounter))
        print("Train loss: {}".format(iterTrainLoss/outputTrainCounter))


    return net

def main():
    args = parse_args()

    with open(args.net) as jsonFile:
        jsonNetSpecification = json.load(jsonFile, object_pairs_hook=OrderedDict)
    with open(args.train) as jsonFile:
        jsonTrainSpecification = json.load(jsonFile, object_pairs_hook=OrderedDict)

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

    net = CreateNet(netSpecification)
    net = TrainNet(trainSpecification, netSpecification, net)

    if args.output is not None:
        f = open(args.output + ".bn", "wb")
        pickle.dump(net, f)
        f.close()


if __name__ == "__main__":
    main()
