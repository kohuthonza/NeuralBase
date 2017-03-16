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
import cPickle

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
    parser.add_argument('-i', '--input',
                        type=str,
                        help='Name of net to be load')

    args = parser.parse_args()

    return args

def parseJSONNetSpecification(jsonNetSpecification):
    netSpecification = {}
    layersProperties = ['type', 'eps', 'numberOfNeurons', 'bias', 'kernelSize', 'numberOfKernels', 'stride']
    layersSpecifications = []

    if 'grayscale' in jsonNetSpecification:
        if jsonNetSpecification['grayscale'] == 'True' or jsonNetSpecification['grayscale'] == 'true':
            netSpecification['grayscale'] = 0
        else:
            netSpecification['grayscale'] = 1
    else:
        netSpecification['grayscale'] = 1

    if 'layers' in jsonNetSpecification:
        for jsonLayerSpecification in jsonNetSpecification['layers']:
            layer = {}
            for layerProperty in layersProperties:
                if layerProperty in jsonLayerSpecification:
                    layer[layerProperty] = jsonLayerSpecification[layerProperty]

                    if jsonLayerSpecification['type'] == 'FullyConnected':
                        if layerProperty == 'numberOfNeurons':
                            layer[layerProperty] = int(layer[layerProperty])
                        elif layerProperty == 'bias':
                            layer[layerProperty] = float(layer[layerProperty])
                    elif jsonLayerSpecification['type'] == 'BinaryFullyConnected':
                        if layerProperty == 'numberOfNeurons':
                            layer[layerProperty] = int(layer[layerProperty])
                        elif layerProperty == 'bias':
                            layer[layerProperty] = float(layer[layerProperty])
                    elif jsonLayerSpecification['type'] == "BatchNormalization":
                        if layerProperty == 'eps':
                            layer[layerProperty] = float(layer[layerProperty])
                    elif jsonLayerSpecification['type'] == 'Convolutional':
                        if layerProperty == 'kernelSize':
                            layer[layerProperty] = int(layer[layerProperty])
                        elif layerProperty == 'numberOfKernels':
                            layer[layerProperty] = int(layer[layerProperty])
                        elif layerProperty == 'stride':
                            layer[layerProperty] = int(layer[layerProperty])
                else:
                    if layerProperty == 'type':
                        sys.exit("JSON net file doesn't specify type of some layer.")
                    #elif layerProperty == 'numberOfNeurons':
                    #    sys.exit("JSON net file doesn't specify number of neurons in some layer.")
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
    trainSpecificationProperties = ['dataset',
                                    'batchSize',
                                    'meanData', 'scaleData',
                                    'meanLabels', 'scaleLabels',
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
            elif trainProperty == 'meanData' or \
                 trainProperty == 'scaleData' or \
                 trainProperty == 'meanLabels' or \
                 trainProperty == 'scaleLabels' or \
                 trainProperty == 'learningRate' or \
                 trainProperty == 'learnignRateDrop':
                trainSpecification[trainProperty] = float(trainSpecification[trainProperty])

        else:
            if trainProperty == 'dataset':
                sys.exit("JSON train file doesn't specify folder with dataset.")
            elif trainProperty == 'batchSize':
                trainSpecification[trainProperty] = 32
            elif trainProperty == 'numberOfTrainIterations':
                sys.exit("JSON train file doesn't specify number of train iteration.")
            elif trainProperty == 'learningRate':
                sys.exit("JSON train file doesn't specify learning rate.")
            trainSpecification[trainProperty] = None

    return trainSpecification

def PrepareInput(dataLabels, data, batchSize, meanData, scaleData, meanLabels, scaleLabels, grayscale):

    batch = {}
    batch['dataBatch'] = []
    batch['labelsBatch'] = []
    randomIndexes = np.random.randint(len(dataLabels), size=batchSize)
    for index in randomIndexes:
        img = data[index]
        img = img.astype(float)
        if meanData is not None:
            img -= meanData
        if scaleData is not None:
            img = img/scaleData
        batch['dataBatch'].append(img)
        labels = dataLabels[index]
        batch['labelsBatch'].append(labels)
    batch['labelsBatch'] = np.asarray(batch['labelsBatch'])
    if meanLabels is not None:
        batch['labelsBatch'] -= meanLabels
    if scaleLabels is not None:
        batch['labelsBatch'] = batch['labelsBatch']/scaleLabels

    return batch

def CreateNet(netSpecification):
    net = Net.Net()
    net.grayscale = netSpecification['grayscale']
    inputSample = np.zeros((32, 32, 3))
    net.CreateDataShape(inputSample)
    net.CreateLayers(netSpecification['layers'], netSpecification['lossFunction'])
    net.ConectLayers()
    net.InitializeWeights()

    return net

def TrainNet(trainSpecification, net):

    trnData = []
    trnLabels = []
    tstData = []
    tstLabels = []
    for i in range(1,6):
        with open(trainSpecification['dataset'] + 'data_batch_{}'.format(i)) as f:
            data = cPickle.load(f)
        if i == 5:
            tstData = data['data']
            tstLabels = data['labels']
        else:
            trnData.append(data['data'])
            trnLabels.append(data['labels'])
    trnData = np.concatenate(trnData).reshape(-1, 3, 32, 32)
    trnData = np.concatenate([trnData[:,:,:,::-1], trnData[:,:,:,:]])
    trnLabels = np.concatenate(trnLabels)
    trnLabels = np.concatenate([trnLabels, trnLabels])
    tstData = tstData.reshape(-1, 3, 32, 32)
    tstData = np.concatenate([tstData[:,:,:,::-1], tstData[:,:,:,:]])
    tstLabels = np.concatenate([tstLabels, tstLabels])

    trnData = np.rollaxis(trnData, 1, 4)
    tstData = np.rollaxis(tstData, 1, 4)

    trainLabels = np.zeros((trnLabels.shape[0], 10))
    testLabels = np.zeros((tstLabels.shape[0], 10))

    for i in range(0, trainLabels.shape[0]):
        trainLabels[i][trnLabels[i]] = 1
    for i in range(0, testLabels.shape[0]):
        testLabels[i][tstLabels[i]] = 1

    iterTrainLoss = 0
    iterTestLoss = 0
    trainCounter = 0
    outputTrainCounter = 0
    testCounter = 0
    dropFrequencyCounter = 0

    weightsCounter = 0

    accuracy = 0

    for trainCounter in range(trainSpecification['numberOfTrainIterations']):
        batch = PrepareInput(trainLabels,
                            trnData,
                            trainSpecification['batchSize'],
                            trainSpecification['meanData'],
                            trainSpecification['scaleData'],
                            trainSpecification['meanLabels'],
                            trainSpecification['scaleLabels'],
                            net.grayscale)

        net.ForwardPropagation(batch['dataBatch'])
        net.lossLayer.target = batch['labelsBatch']
        net.lossLayer.ForwardOutput()
        net.BackwardPropagation(batch['labelsBatch'])
        net.ActualizeWeights(trainSpecification['learningRate'])

        iterTrainLoss += net.lossLayer.forwardOutput

        trainCounter += 1
        testCounter += 1
        outputTrainCounter += 1


        if weightsCounter == 0:
            fullLayerWeights = np.copy(net.layers[0].weights)
            firstLayerWeights = np.copy(np.sign(net.layers[5].binaryWeights))
            secondLayerWeights = np.copy(np.sign(net.layers[8].binaryWeights))
            lastLayerWeights = np.copy(net.layers[9].weights)
        weightsCounter += 1
        if weightsCounter == 100:
            print np.sum(np.abs(fullLayerWeights - net.layers[0].weights))/float(net.layers[0].weights.size)
            print np.sum(np.abs(firstLayerWeights - np.sign(net.layers[5].binaryWeights)))/2.0
            print np.sum(np.abs(secondLayerWeights - np.sign(net.layers[8].binaryWeights)))/2.0
            print np.sum(np.abs(lastLayerWeights - net.layers[9].weights))/float(net.layers[9].weights.size)
            weightsCounter = 0
        
        if net.lossLayer.layerType == 'SoftMaxCrossEntropy':
            for batchIndex in range(trainSpecification['batchSize']):
                if np.argmax(net.layers[-1].forwardOutput[batchIndex]) == np.argmax(batch['labelsBatch'][batchIndex]):
                    accuracy += 1


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
                if net.lossLayer.layerType == 'SoftMaxCrossEntropy':
                    print("Accuracy: {}".format(float(accuracy)/(trainSpecification['trainOutputFrequency'] * trainSpecification['batchSize'])))
                accuracy = 0

        if trainSpecification['testOutputFrequency'] is not None:
            if testCounter == trainSpecification['testOutputFrequency']:
                testCounter = 0
                accuracy = 0
                print("")
                print("")
                print('********************************************')
                print('Testing net...')
                for counter in range(trainSpecification['numberOfTestIterations']):
                    batch = PrepareInput(testLabels,
                                        tstData,
                                        trainSpecification['batchSize'],
                                        trainSpecification['meanData'],
                                        trainSpecification['scaleData'],
                                        trainSpecification['meanLabels'],
                                        trainSpecification['scaleLabels'],
                                        net.grayscale)

                    net.ForwardPropagation(batch['dataBatch'])
                    net.lossLayer.target = batch['labelsBatch']
                    net.lossLayer.ForwardOutput()
                    iterTestLoss += net.lossLayer.forwardOutput

                    if net.lossLayer.layerType == 'SoftMaxCrossEntropy':
                        for batchIndex in range(trainSpecification['batchSize']):
                            if np.argmax(net.layers[-1].forwardOutput[batchIndex]) == np.argmax(batch['labelsBatch'][batchIndex]):
                                accuracy += 1
                print('____________________________________________')
                print('Last batch from training set:')
                print("Net: {}".format(net.layers[-1].forwardOutput[batchIndex]))
                print("Label: {}".format(batch['labelsBatch'][batchIndex]))
                print('{} test iteration DONE.'.format(trainSpecification['numberOfTestIterations']))
                if net.lossLayer.layerType == 'SoftMaxCrossEntropy':
                    print("Accuracy: {}".format(float(accuracy)/(trainSpecification['numberOfTestIterations'] * trainSpecification['batchSize'])))
                else:
                    print("Test loss: {}".format(iterTestLoss/trainSpecification['numberOfTestIterations']))
                print('********************************************')
                print("")
                iterTestLoss = 0
                accuracy = 0

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

    if args.input is not None:
        f = open(args.input + ".bn", "rb")
        net = pickle.load(f)
        f.close()
    else:
        netSpecification = parseJSONNetSpecification(jsonNetSpecification)
        net = CreateNet(netSpecification)

    trainSpecification = parseJSONTrainSpecification(jsonTrainSpecification)
    net = TrainNet(trainSpecification, net)

    if args.output is not None:
        f = open(args.output + ".bn", "wb")
        pickle.dump(net, f)
        f.close()


if __name__ == "__main__":
    main()
