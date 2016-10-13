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
    <td>JSON definition of the net.</td>
  </tr>
  <tr>
    <td><b>--train, -t</b></td> 
    <td>JSON definition of the training process.</td>
  </tr>
  <tr>
    <td><b>--output, -o</b></td> 
    <td>Name of the net to be saved.</td>
  </tr>
   <tr>
    <td><b>--input, -i</b></td> 
    <td>Name of the net to be load.</td>
  </tr>
</table>
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
    <td>Loss function of the net.</td>
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
    <td>Number of data to be procced at once.</td>
  </tr>
  <tr>
    <td><b>meanData (optional, float)</b></td> 
    <td>Value substracted from images.</td>
  </tr>
  <tr>
    <td><b>scaleData (optional, float)</b></td> 
    <td>Data are multiplied by scaleData.</td>
  </tr>
  <tr>
    <td><b>meanLabels (optional, float)</b></td> 
    <td>Value substracted from labels.</td>
  </tr>
  <tr>
    <td><b>scaleLabels (optional, float)</b></td> 
    <td>Labels are multiplied by scaleLabels.</td>
  </tr>
  <tr>
    <td><b>learningRate (float)</b></td> 
    <td>Learning rate for gradient descent.</td>
  </tr>
  <tr>
    <td><b>learningRateDrop (float)</b></td> 
    <td>Learning rate is multiply by learningRateDrop every dropFrequency.</td>
  </tr>
  <tr>
    <td><b>dropFrequency (train iterations)</b></td> 
    <td>Define how often is learning rate decreased.</td>
  </tr>
  <tr>
    <td><b>numberOfTrainIterations (integer)</b></td> 
    <td>Number of iterations.</td>
  </tr>
  <tr>
    <td><b>trainOutputFrequency (iteration)</b></td> 
    <td>How often is train loss outputed.</td>
  </tr>
  <tr>
    <td><b>numberOfTestIterations (integer)</b></td> 
    <td>Number of test iterations.</td>
  </tr>
  <tr>
    <td><b>testOutputFrequency (iteration)</b></td> 
    <td>How often is test procced and outputed.</td>
  </tr>
</table>




