# DenseNet_lite

This implements the DenseNet architecture introduced in [Densely Connected Convolutional Network](http://arxiv.org/abs/1608.06993).The original Torch implementation can be found at https://github.com/liuzhuang13/DenseNet, and please find more details about DenseNet there. The only difference here is that we write a customed container "DenseLayer.lua" to implement the dense connections in a more memory efficient way. This leads to **~25% reduction** in memory consumption during training, while keeps the accuracy and training time the same. 

##Usage 
0. Install Torch ResNet (https://github.com/facebook/fb.resnet.torch) following the instructions there. To reduce memory consumption, we recommend to install the [optnet](https://github.com/fmassa/optimize-net) package. 
1. Add the files ```densenet_lite.lua``` and ```DenseLayer.lua``` to the folder models/.
2. Change the learning rate schedule at function learningRate() in ```train.lua``` (line 171/173),
from 
```decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0```
to 
 ```decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0 ```
 
3. Train a DenseNet (L=40, k=12) on CIFAR-10+ using

```
th main.lua -netType densenet_lite -depth 40 -dataset cifar10 -batchSize 64 -nEpochs 300 -optnet true
``` 



