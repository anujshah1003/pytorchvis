import os
import torch
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np

class VisualizeLayers(object):
    '''
    A class to visualize intermediate layer outputs
    '''
    
    def __init__(self,model,layers='all'):
        
        self.model = model
        self.layers = layers
        self.interm_output = {}
        self.hook_layers(self.model)
        if not os.path.exists('output_imgs'):
            os.makedirs('output_imgs')
        
    def __get_activation(self,name):
        def hook(model, input, output):
            self.interm_output[name] = output.detach()
        return hook
        
    def hook_layers(self,model,prev_name=None):
        '''
        function to hook different layers
        Args:
            model - () - pytorch model or layers
            prev_name -(string or None) name of the prev layer(father layer)
        '''
        for name, layer in model._modules.items():
            layer_type_str=str(type(layer)).split('.')
            layer_type = layer_type_str[-2]
            layer_sub_type = layer_type_str[-1][:-2]
            if prev_name is None:
                layer_name=name+'_{}_{}'.format(layer_type,layer_sub_type)
                print (layer_name)
            else:
                layer_name = '{}.{}_{}_{}'.format(prev_name,name,layer_type,layer_sub_type)
                print ('\t',layer_name)
            if self.layers=='all':
                layer.register_forward_hook(self.__get_activation(layer_name))
            else:       
                if layer_type in self.layers:
                    layer.register_forward_hook(self.__get_activation(layer_name))
                    
            if layer._modules.items() is not None:
                self.hook_layers(layer,prev_name=name)
    
    def get_interm_output(self):
        '''
        function to get the intermediate layers(layers which were hooked) output
        
        returns:
            self.interim_output: (dict)- a dictionary of intermediate layer 
            outputs which were hooked
        '''
        return self.interm_output
        
    def get_saved_layer_names(self):
        '''
        function to get the intermediate layer names whose output are saved
        Returns:
            self.interm_output_keys(): (dict_keys)
        '''
        return self.interm_output.keys()
    
    def plot_featuremaps(self,featuremaps,name='featuremaps',color_map='gray',savefig=False,figsize=12):
        '''
        function to plot the feature maps of an intermediate layer
        Args:
           featuremaps: (torch.tensor) - a tensor of shape [1,64,55,55] 
                       representing (Batch_size, num_featuremaps,
                       height of each featuremap,width of each featuremap)
            name: (string) - name of the feature map
            color_map: (string) - 'gray' or 'viridis'
            savefig: (Bool) - True or False , whether or not you want to save 
                      the fig
            figsize: (int) - figure size in th form of (figsize,figsize)
        '''
        featuremaps=featuremaps.squeeze()
        num_feat_maps=featuremaps.size(0)
        subplot_num=int(np.ceil(np.sqrt(num_feat_maps)))
    #    subplot_r = int(num_feat_maps/8)
    #    subplot_c = 8
        fig = plt.figure()
        plt.figure(figsize=(figsize,figsize))
        for idx,f_map in enumerate(featuremaps):
            #print(filt[0, :, :])
            plt.subplot(subplot_num,subplot_num, idx + 1)
    #        plt.subplot(subplot_r,subplot_c, idx + 1)
            plt.imshow(f_map,cmap=color_map)
            plt.axis('off')
        fig.show()
        plt.savefig("output_imgs/{}".format(name) + '.jpg')

#%%        
if __name__=='__main__':  
    
    # get the device on which the model is going to run
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
    
    vis.plot_featuremaps(interm_output['features.0_conv_Conv2d'],name='noise_inpt_fmap-1',color_map='gray',savefig=True)
    
    


    
