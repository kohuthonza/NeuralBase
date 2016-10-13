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
    <td><b>activationFunction</b></td> 
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

 "dataMean":"127.0",
 "dataScale":"127.0",

 "learningRate":"0.001",

 "numberOfTrainIterations":"500",
 "trainOutputFrequency":"20",
 "numberOfTestIterations":"10",
 "testOutputFrequency":"100"}
```

