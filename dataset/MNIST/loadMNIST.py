from mnist import MNIST
import numpy as np
import cv2
import itertools
import os
import sys
import shutil

mndata = MNIST("")
mndata.load_training()
mndata.load_testing()

dataMNIST = [mndata.train_images, mndata.test_images]
labelsMNIST = [mndata.train_labels, mndata.test_labels]
datasetNames = ['train', 'test']

for data, labels, datasetName in itertools.izip(dataMNIST, labelsMNIST, datasetNames):
    f = open(datasetName + '.txt', 'w')
    if os.path.isdir(datasetName):
        shutil.rmtree(datasetName)
    os.mkdir(datasetName)
    counter = 0
    for image, label in itertools.izip(data, labels):
        newLabel = np.zeros(10, dtype=np.float)
        newLabel[int(label)] = 1.0
        newLabel = [str(tmp) for tmp in newLabel]
        newLabel = " ".join(newLabel)
        imageName = datasetName + '_' +str(counter) + '.png'
        f.write(imageName + " " + newLabel + '\n')
        image = np.asarray(image).reshape(28, 28)
        cv2.imwrite(os.path.join(datasetName,imageName), image)
        counter += 1
    f.close()
