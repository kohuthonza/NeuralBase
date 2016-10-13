<div align="center">
  <img src="http://www.stud.fit.vutbr.cz/~xkohut08/logoNeuralBase.png"><br><br>
</div>

Basic implementation of neural networks.

<h2>
Arguments
</h2>
<h3>
CreateNet.py
</h3>
<table style="width:50%">
   <tr>
    <th>Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>--net, -n</b></td>
    <td>JSON definition of net.</td>
  </tr>
  <tr>
    <td><b>--train, -t</b></td> 
    <td>JSON definition of training process.</td>
  </tr>
  <tr>
    <td><b>--output, -o</b></td> 
    <td>Name of net to be saved.</td>
  </tr>
   <tr>
    <td><b>--input, -i</b></td> 
    <td>Name of net to be load.</td>
  </tr>
</table>

<h3>
Usage
</h3>

<h4>
Create new net
</h4>

python CreateNet.py --net net.json --train --train.json --output newNet.nb

<h4>
Finetuning
</h4>

python CreateNet.py --train train.json --input net.nb --output newNet.nb

<br>

<h2>
JSON definitions
</h2>

<h3>
Net
</h3>

```json
{"inputSample":"/dir/trainData/train_0.png",
 "grayscale":"False",
 "layers":[
          {"type":"FullyConnected", 
           "numberOfNeurons":"100",
           "bias":"0.1", 
           "activationFunction":"Sigmoid"},
          {"type":"FullyConnected", 
          "numberOfNeurons":"3", 
          "activationFunction":"SoftMax"} 
          ],
 "lossFunction":"SoftMaxCrossEntropy"}
```
<table style="width:50%">
   <tr>
    <th>Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>inputSample</b></td>
    <td>Path to train file.</td>
  </tr>
  <tr>
    <td><b>grayscale</b> (optinal, default=False)</td> 
    <td>Load images in grayscale.</td>
  </tr>
  <tr>
    <td><b>layers</b></td> 
    <td>List of layers definitions.</td>
  </tr>
   <tr>
    <td><b>lossFunction</b></td> 
    <td>Loss function of net.</td>
  </tr>
</table>

<h4>
Types of layers
</h4>

<table style="width:50%">
  <tr>
    <td colspan="2">Fully connected</td>
  </tr>
  <tr>
    <td><b>type</b></td>
    <td>FullyConnected</td>
  </tr>
  <tr>
    <td><b>numberOfNeurons</b></td> 
    <td>Number of neurons in layer (integer).</td>
  </tr>
  <tr>
    <td><b>bias</b> (optional)</td> 
    <td>Bias (float).</td>
  </tr>
   <tr>
    <td><b>activationFunction</b> (optional)</td> 
    <td>Activation function (Sigmoid, SoftMax).</td>
  </tr>
</table>

<h3>
Train 
</h3>
```json
{"trainData":"/dir/trainData",
 "trainLabels":"/dir/train.txt",
 "testData":"/dir/testData",
 "testLabels":"/dir/test.txt",

 "batchSize":"32",

 "meanData":"127.0",
 "scaleData":"127.0",
 "meanLabels":"0.5",
 "scaleLabels":"2.0",
 
 "learningRate":"0.001",
 "learningRateDrop":"0.5",
 "dropFrequency":"1000",

 "numberOfTrainIterations":"10000",
 "trainOutputFrequency":"20",
 "numberOfTestIterations":"10",
 "testOutputFrequency":"100"}
```

<table style="width:50%">
   <tr>
    <th>Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>trainData</b></td>
    <td>Path to folder with train data.</td>
  </tr>
  <tr>
     <td><b>trainLabels</b></td>
    <td>Path to file with train labels.</td>
  </tr>
  <tr>
    <td><b>testData</b></td>
    <td>Path to folder with test data.</td>
  </tr>
  <tr>
     <td><b>testLabels</b></td>
    <td>Path to file with test labels.</td>
  </tr>
  <tr>
    <td><b>batchSize</b></td> 
    <td>Number of data to be procced at once (integer).</td>
  </tr>
  <tr>
    <td><b>meanData</b> (optional)</td> 
    <td>Value substracted from images (float).</td>
  </tr>
  <tr>
    <td><b>scaleData</b> (optional)</td> 
    <td>Data are multiplied by scaleData (float).</td>
  </tr>
  <tr>
    <td><b>meanLabels</b> (optional)</td> 
    <td>Value substracted from labels (float).</td>
  </tr>
  <tr>
    <td><b>scaleLabels</b> (optional)</td> 
    <td>Labels are multiplied by scaleLabels.</td>
  </tr>
  <tr>
    <td><b>learningRate</b></td> 
    <td>Learning rate for gradient descent (float).</td>
  </tr>
  <tr>
    <td><b>learningRateDrop</b> (optional)</td> 
    <td>Learning rate is multiply by learningRateDrop every dropFrequency (float).</td>
  </tr>
  <tr>
    <td><b>dropFrequency</b> (optional)</td> 
    <td>Define how often is learning rate decreased (train iterations).</td>
  </tr>
  <tr>
    <td><b>numberOfTrainIterations</b></td> 
    <td>Number of iterations (integer).</td>
  </tr>
  <tr>
    <td><b>trainOutputFrequency</b> (optional)</td> 
    <td>How often is train loss outputed (train iterations).</td>
  </tr>
  <tr>
    <td><b>numberOfTestIterations</b> (optional)</td> 
    <td>Number of test iterations (integer).</td>
  </tr>
  <tr>
    <td><b>testOutputFrequency</b> (optional)</td> 
    <td>How often is test procced and outputed  (train iterations).</td>
  </tr>
</table>
<br>

<h2>
Datasets
</h2>

<h3>
MNIST
</h3>
Require [python-mnist](https://github.com/sorki/python-mnist) to be installed. Run loadMNIST.py in directory with [MNIST](http://yann.lecun.com/exdb/mnist/) ubyte files.
 
<h4>
Usage
</h4>
python loadMNIST.py

<h3>
Basic objects
</h3>
Generate basic objects of random color. 

<h4>
Usage
</h4>
python GeneratorBasicObjects.py --size 32 --super-size 128 --min-size 80 --train-size 20000 --test-size 2000 --type 
python GeneratorBasicObjects.py --size 32 --super-size 128 --min-size 80 --train-size 20000 --test-size 2000 --color

<h3>
Try it yourself!
</h3>

Create folder of images for training and testing. Every image has to be labeled. Each line of label file contains name of image and it's label. Labels for net train with euclidean distance loss function are values. Labels for net train with softmax activation function and cross entropy loss function are zero vectors with one specifies class.
<br>
<br>
Sample of label file for training with euclidean distance loss function:<br>
```
testRGB_0.png 0.419608 0.541176 0.011765<br>
testRGB_1.png 0.800000 0.376471 0.109804<br>
testRGB_2.png 0.070588 0.349020 0.623529<br>
testRGB_3.png 0.262745 0.376471 0.015686<br>
```
<br>
Sample of label file for training with softmax and cross entropy loss function (3 classes):<br>
```
test_0.png 0.000000 1.000000 0.000000<br>
test_1.png 0.000000 1.000000 0.000000<br>
test_2.png 0.000000 1.000000 0.000000<br>
test_3.png 0.000000 0.000000 1.000000
```




