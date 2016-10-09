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
    parser.add_argument('-ei', '--test-iteration',
                        required=True,
                        type=int,
                        help='Number of test iteration')
    parser.add_argument('-eo', '--test-frequency',
                        required=True,
                        type=int,
                        help='How often run test in train iteration')
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


    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open(args.train_dataset_file) as f:
        trainData = f.read().splitlines()

    for dataIndex in range(0, len(trainData)):
        trainData[dataIndex] = trainData[dataIndex].split()

    if args.gray_scale:
        imgColor = 0
    else:
        imgColor = 1

    batchData = []
    batchLabels = []
    testCounter = 0

    net = Net.Net()
    inputSample = cv2.imread(os.path.join(args.train_dataset_directory + trainData[0][0]), imgColor)
    net.CreateDataShape(inputSample, args.batch_size)
    layersProperties = [(100, 'Sigmoid'), (100, 'Sigmoid'), (3, None)]
    net.CreateLayers(layersProperties, 'EuclideanDistance')
    net.ConectLayers()
    net.InitializeWeights()


    for counter in range(0, args.train_iteration):
        randomIndexes = np.random.randint(len(trainData), size=args.batch_size)
        for index in randomIndexes:
            batchData.append(cv2.imread(os.path.join(args.train_dataset_directory + trainData[index][0]), imgColor))
            labels = trainData[index][1:]
            batchLabels.append([float(label) for label in labels])
        batchLabels = np.asarray(batchLabels).reshape(args.batch_size, 1, -1, 1)

        net.ForwardPropagation(batchData)
        net.lossLayer.LossOutput(batchLabels)
        print("loss: {}".format(net.lossLayer.lossOutput))

        batchData = []
        batchLabels = []


        if counter != 0:
            if counter % 100 == 0:
                print('{} train iteration DONE.'.format(counter))



        counter += 1
        testCounter += 1

        #if testCounter == args.test-frequency:


    print('{} train iteration DONE.'.format(counter))

if __name__ == "__main__":
    main()
