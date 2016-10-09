import cv2
import sys
import os
import argparse
import numpy as np
import Net

def parse_args():
    print( ' '.join(sys.argv))

    parser = argparse.ArgumentParser(epilog="NeuralBase")

    parser.add_argument('-n', '--net',
                        type=str,
                        help='Net properties')
    parser.add_argument('-ti', '--train-iteration',
                        required=True,
                        type=int,
                        help='Number of train iteration')
    parser.add_argument('-to', '--train-frequency',
                        required=True,
                        type=int,
                        help='How often output train loss')
    parser.add_argument('-ei', '--test-iteration',
                        required=True,
                        type=int,
                        help='Number of test iteration')
    parser.add_argument('-eo', '--test-frequency',
                        required=True,
                        type=int,
                        help='How often run test and output test loss in train iteration')
    parser.add_argument('-bs', '--batch-size', default=32,
                        type=int)
    parser.add_argument('-tf', '--train-dataset-file',
                        required=True,
                        help='File with names and labels of data for train.')
    parser.add_argument('-td', '--train-dataset-directory',
                        required=True,
                        help='Directory with data for train.')
    parser.add_argument('-ef', '--test-dataset-file',
                        required=True,
                        help='File with names and labels of data for test.')
    parser.add_argument('-ed', '--test-dataset-directory',
                        required=True,
                        help='Directory with data for test.')
    parser.add_argument('-m', '--mean',
                        type=float,
                        help='Substract mean from data.')
    parser.add_argument('-s', '--scale',
                        type=float,
                        help='Scale images.')
    parser.add_argument('-g', '--gray-scale',
                        action='store_true',
                        help='Load images in grayscale.')
    parser.add_argument('-ea', '--test-all',
                        action='store_true',
                        help='Test net on all data for test.')


    args = parser.parse_args()

    return args

def IterNet(net, data, dataDirectory, batchSize, imgColor):
    batchData = []
    batchLabels = []
    randomIndexes = np.random.randint(len(data), size=batchSize)
    for index in randomIndexes:
        batchData.append(cv2.imread(os.path.join(dataDirectory + data[index][0]), imgColor))
        labels = data[index][1:]
        batchLabels.append([float(label) for label in labels])
    batchLabels = np.asarray(batchLabels)

    net.ForwardPropagation(batchData)
    net.lossLayer.LossOutput(batchLabels)

def main():
    args = parse_args()

    with open(args.train_dataset_file) as f:
        trainData = f.read().splitlines()
    with open(args.test_dataset_file) as f:
        testData = f.read().splitlines()

    for dataIndex in range(0, len(trainData)):
        trainData[dataIndex] = trainData[dataIndex].split()
    for dataIndex in range(0, len(testData)):
        testData[dataIndex] = testData[dataIndex].split()

    if args.gray_scale:
        imgColor = 0
    else:
        imgColor = 1

    net = Net.Net()
    inputSample = cv2.imread(os.path.join(args.train_dataset_directory + trainData[0][0]), imgColor)
    net.CreateDataShape(inputSample)
    layersProperties = [('FullyConnected', 100, 'Sigmoid', 0.0), ('FullyConnected', 100, 'Sigmoid', 0.0), ('FullyConnected', 3, None, None)]
    net.CreateLayers(layersProperties, 'EuclideanDistance', inputSample)
    net.ConectLayers()
    net.InitializeWeights()


    iterTrainLoss = 0
    iterTestLoss = 0
    trainCounter = 0
    testCounter = 0

    for counter in range(0, args.train_iteration):
        IterNet(net, trainData, args.train_dataset_directory, args.batch_size, imgColor)
        iterTrainLoss += net.lossLayer.lossOutput

        counter += 1
        testCounter += 1
        trainCounter += 1

        if trainCounter == args.train_frequency:
            trainCounter = 0
            print('{} train iteration DONE.'.format(counter))
            print("Train loss: {}".format(iterTrainLoss/args.train_frequency))
            iterTrainLoss = 0

        if testCounter == args.test_frequency:
            testCounter = 0
            for counter in range(0, args.test_iteration):
                IterNet(net, testData, args.test_dataset_directory, args.batch_size, imgColor)
                iterTestLoss += net.lossLayer.lossOutput
            print('{} test iteration DONE.'.format(args.test_iteration))
            print("Test loss: {}".format(iterTestLoss/args.test_iteration))
            iterTestLoss = 0

    if iterTrainLoss != 0:
        print('{} train iteration DONE.'.format(counter))
        print("Train loss: {}".format(iterTrainLoss/100.0))

    #TODO
    if args.test_all:
        for counter in range(0, len(testData)):
            IterNet(net, testData, args.test_dataset_directory, args.batch_size, imgColor)
            iterTestLoss += net.lossLayer.lossOutput
        print('{} test iteration DONE.'.format(args.test_iteration))
        print("Test loss: {}".format(iterTestLoss/args.test_iteration))


if __name__ == "__main__":
    main()
