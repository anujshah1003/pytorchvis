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
  vis.get_saved_layer_names()
  vis.plot_featuremaps(interm_output['features.0_conv_Conv2d'],name='fmaps',savefig=True)

```
#### the 64 featurmap from the first conv layer with a random input
![](https://github.com/anujshah1003/pytorchvis/blob/master/pytorchvis/output_imgs/noise_inpt_fmap-1.jpg)

## Naming convention of the saved intermediate layers

layer name and its sublayers are separated by . (dot) for e.g
```
  features.0  - layer 0 is a sub layer of layer features
  features.feat1.conv1 - conv1 is the layer name whic is a sub layer of feat1 which further is the sub layer of features
```
the class and type of layer is given by underscore after the layer name for e.g.
```
features._feat1.conv1_conv_Conv2D - this layer name is conv1 and its is from class conv and its type is Conv2d
features.0_conv_Conv2D - this layer name is 0 and its from class conv and its type is Conv2D
```


## Reference:

ptrblck hook function from - https://discuss.pytorch.org/t/visualize-feature-map/29597

Library Motivation - https://github.com/sksq96/pytorch-summary
