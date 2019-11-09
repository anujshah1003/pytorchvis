# pytorchvis
A library to visualize CNN in PyTorch.
## Installation

```
pip install pytorchvis
git clone https://github.com/anujshah1003/pytorchvis
```
## Usage
```
  from pytorch.visualize_layers import VisualizeLayers

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

### Reference:

ptrblck hook function from - https://discuss.pytorch.org/t/visualize-feature-map/29597

Library Motivation - https://github.com/sksq96/pytorch-summary
