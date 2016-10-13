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
    <td>--net, -n</td>
    <td>JSON definition of the net.</td>
  </tr>
  <tr>
    <td>--train, -t</td> 
    <td>JSON definition of the training process.</td>
  </tr>
  <tr>
    <td>--output, -o</td> 
    <td>Name of the net to be saved.</td>
  </tr>
   <tr>
    <td>--input, -i</td> 
    <td>Name of the net to be load.</td>
  </tr>
</table>
<br>

<h3>
JSON definitions
</h3>
<h4>
Net
</h4>

```json
{"inputSample":"/dir/data",
 "grayscale":"False",
 "layers":[
          {"type":"FullyConnected", 
           "numberOfNeurons":"100",
           "bias":"0.1", 
           "activationFunction":"Sigmoid"},
          ],
 "lossFunction":"SoftMaxCrossEntropy"}
```


