# pytorchvis
A library to visualize CNN in PyTorch.
## Installation

```
pip install pytorchvis
git clone https://github.com/anujshah1003/pytorchvis
```
## Usage
```
  from pytorchvis.visualize_layers import VisualizeLayers

  # create an object of VisualizeLayers and initialize it with the model and 
  # the layers whose output you want to visualize        
  vis = VisualizeLayers(model,layers='conv')

  # pass the input and get the output
  output = model(x)

  # get the intermediate layers output which was passed during initialization
  interm_output = vis.get_interm_output()

  # plot the featuremap of the layer which you want,
  vis.plot_featuremaps(interm_output[layer_name],name='fmaps',savefig=True)
  
```
##### interm_output is the dictionary which stores the intermediate putput. It's keys are the layer names and its values are the respective output of the intermediate layers.

## Example
### Using Pretrained Alexnet
```
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # load the Pytorch model
  model = models.alexnet(pretrained=True).to(device)
  # create an object of VisualizeLayers and initialize it with the model and 
  # the layers whose output you want to visualize

  vis = VisualizeLayers(model,layers='conv')
  # load the input
  x = torch.randn([1,3,224,224]).to(device)
  # pass the input and get the output
  output = model(x)
  # get the intermediate layers output which was passed during initialization
  interm_output = vis.get_interm_output()

  # plot the featuremap of the layer which you want, to see what are the layers
  # saved simply call vis.get_saved_layer_names
  vis.get_saved_layer_names() # the key values of the dictionary interm_output
  vis.plot_featuremaps(interm_output['features.0_conv_Conv2d'],name='fmaps',savefig=True)

```
#### the 64 featurmap from the first conv layer with a random input
![](https://github.com/anujshah1003/pytorchvis/blob/master/pytorchvis/output_imgs/noise_inpt_fmap-1.jpg)

## Naming convention of the saved intermediate layers

layer name and its sublayers are separated by . (dot) for e.g
```
  features.0  - layer 0 is a sub layer of layer features.
  features.feat1.conv1 - conv1 is the layer name whic is a sub layer of feat1 which further
  is the sub layer of features.
```
the class and type of layer is given by underscore after the layer name for e.g.
```
features._feat1.conv1_conv_Conv2D - this layer name is conv1 and its is from class conv and its type is Conv2d
features.0_conv_Conv2D - this layer name is 0 and its from class conv and its type is Conv2D
```
Another example, say you have an alex net model and then you print the model to see all the layers
```
  model = models.alexnet(pretrained=True)
  print(model)
  
  # this are your layers in the Alexnet model
  
  AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace)
    (3): Dropout(p=0.5)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
  
 Now if you call 
 vis = VisualizeLayers(model,layers='conv')
 
 The layer names that are registered as below and depending on the layers argument in 
 VisualizeLayers(layers='all' or layers='conv' or layers='activation'), the respective intermediate 
 layers output will be saved in interm_output = vis.get_interm_output() 
 
 features_container_Sequential
         features.0_conv_Conv2d
         features.1_activation_ReLU
         features.2_pooling_MaxPool2d
         features.3_conv_Conv2d
         features.4_activation_ReLU
         features.5_pooling_MaxPool2d
         features.6_conv_Conv2d
         features.7_activation_ReLU
         features.8_conv_Conv2d
         features.9_activation_ReLU
         features.10_conv_Conv2d
         features.11_activation_ReLU
         features.12_pooling_MaxPool2d
avgpool_pooling_AdaptiveAvgPool2d
classifier_container_Sequential
         classifier.0_dropout_Dropout
         classifier.1_linear_Linear
         classifier.2_activation_ReLU
         classifier.3_dropout_Dropout
         classifier.4_linear_Linear
         classifier.5_activation_ReLU
         classifier.6_linear_Linear
```
To get the saved layer names
```
vis.get_saved_layer_names()

['features.0_conv_Conv2d',
 'features.3_conv_Conv2d',
 'features.6_conv_Conv2d',
 'features.8_conv_Conv2d',
 'features.10_conv_Conv2d']
 
```

## Reference:

ptrblck hook function from - https://discuss.pytorch.org/t/visualize-feature-map/29597

Library Motivation - https://github.com/sksq96/pytorch-summary
