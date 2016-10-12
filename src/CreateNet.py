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
                        help='JSON specification of net')
    parser.add_argument('-t', '--train',
                        type=str,
                        help='JSON specification of train process')

    args = parser.parse_args()

    return args

def parseJSONNetSpecification(jsonNetSpecification):

    jsonNetSpecification = json.loads(jsonNetSpecification)

    layersProperties = ['type', 'numberOfNeurons', 'bias', 'activationFunction']
    layersSpecifications = []

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

    netSpecification = {}
    netSpecification['layers'] = layersSpecifications

    if 'lossFunction' in jsonNetSpecification:
        netSpecification['lossFunction'] = jsonNetSpecification['lossFunction']
    else:
        sys.exit("JSON net file doesn't specify loss layer.")


    return netSpecification

def parseJSONTrainSpecification(jsonTrainSpecification):

    jsonTrainSpecification = json.loads(jsonNetSpecification)

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
            if trainProperty == 'batchSize' or
               trainProperty == 'dropFrequency' or
               trainProperty == 'numberOfTrainIterations' or
               trainProperty == 'numberOfTestIterations' or
               trainProperty == 'trainOutputFrequency' or
               trainProperty == 'testFrequency':
                trainSpecification[trainProperty] = int(trainSpecification[trainProperty])
            elif trainProperty == 'mean' or
                 trainProperty == 'scale' or
                 trainProperty == 'learningRate' or
                 trainProperty == 'learnignRateDrop':
                trainSpecification[trainProperty] = float(trainSpecification[trainProperty])
            elif layerProperty == 'grayscale':
                if trainSpecification[trainProperty] == 'True' or trainSpecification[trainProperty] == 'true':
                    trainSpecification[trainProperty] = 0
                else:
                    trainSpecification[trainProperty] = 1
        else:
            if trainProperty == 'trainData':
                sys.exit("JSON train file doesn't specify folder with training data.")
            elif trainProperty == 'trainLabels':
                sys.exit("JSON train file doesn't specify file with training labels.")
            elif trainProperty = 'batchSize':
                trainSpecification[trainProperty] = 32
            elif trainProperty = 'numberOfTrainIterations':
                sys.exit("JSON train file doesn't specify number of train iteration.")
            elif trainProperty = 'learningRate':
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

def CreateNet(netSpecification, inputSample):

    net = Net.Net()
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
    testCounter = 0

    for counter in range(0, trainSpecification['numberOfTrainIterations']):

        batch = PrepareInput(labels['trainLabels'],
                            jsonTrainSpecification['trainData'],
                            trainSpecification['bacthSize'],
                            trainSpecification['mean'],
                            trainSpecification['scale'],
                            grayscale)

        net.ForwardPropagation(bacth['dataBatch'])
        net.lossLayer.LossOutput(batch['labelsBatch'])
        net.BackwardPropagation(batch['labelsBatch'])
        net.ActualizeWeights(trainSpecification['learningRate']))

        iterTrainLoss += net.lossLayer.lossOutput

        counter += 1
        testCounter += 1
        trainCounter += 1

        if trainSpecification['trainOutputFrequency'] is not None:
            if trainCounter == trainSpecification['trainOutputFrequency']:
                trainCounter = 0
                print('{} train iteration DONE.'.format(counter))
                print("Train loss: {}".format(iterTrainLoss/args.train_frequency))
                iterTrainLoss = 0

        if testCounter == args.test_frequency:
            testCounter = 0
            accuracy = 0
            for counter in range(0, args.test_iteration):

                data = PrepareInput(testData, args.test_dataset_directory, args.batch_size, imgColor)
                batchData = data[0]
                batchLabels = data[1]

                net.ForwardPropagation(batchData)
                net.lossLayer.LossOutput(batchLabels)
                iterTestLoss += net.lossLayer.lossOutput

                for batchIndex in range(0, args.batch_size):
                    if np.argmax(net.fullyConnectedLayers[-1].forwardOutput[batchIndex]) == np.argmax(batchLabels[batchIndex]):
                        accuracy += 1
                    print("Net: {}".format(net.fullyConnectedLayers[-1].forwardOutput[batchIndex]))
                    print("Label: {}".format(batchLabels[batchIndex]))

            print('{} test iteration DONE.'.format(args.test_iteration))
            print("Test loss: {}".format(iterTestLoss/args.test_iteration))
            print("Accuracy: {}".format(float(accuracy)/(args.test_iteration * args.batch_size)))
            iterTestLoss = 0

    if iterTrainLoss != 0:
        print('{} train iteration DONE.'.format(counter))
        print("Train loss: {}".format(iterTrainLoss/iterTrainLoss))


def main():
    args = parse_args()

    inputSample = cv2.imread(os.path.join(args.train_dataset_directory + trainData[0][0]), imgColor)

    netSpecification = parseJSONNetSpecification(args.net)
    net = CreateNet(netSpecification, inputSample)

    trainSpecification = parseJSONTrainSpecification(args.train)
    net = TrainNet(trainSpecification, net)



if __name__ == "__main__":
    main()
