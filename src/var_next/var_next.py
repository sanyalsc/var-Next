import torch
from torch import nn
import json

class varNext(torch.nn.Module):
    """Variational Autoencoder definition.
    
    cfg definitions:

    layers
    - encoder
        - {conv
            { in_channel
              out_channel
              kernel_size
              kwargs
                { stride
                  padding
                  dilation}
            }
           activation:
                (one of 'leakyrelu', 'relu')
        .
        .
        .

    """
    def __init__(self, cfg_file):
        if type(cfg_file)!= dict:
            with open(cfg_file,'r') as f:
                cfg_file = json.load(f)

        # Build encoder
        enc_layers = []
        for layer in cfg_file['layers']['encoder']:
            conv = layer['conv']
            if layer['activation']=='leakyrelu':
                activation = nn.LeakyReLU()
            else:
                activation = nn.ReLU()

            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(conv['in_channels'],conv['out_channels'],conv['kernel_size'],**conv),
                    nn.BatchNorm2d(conv['out_channels']),
                    activation)
            )
        
        self.encoder = nn.Sequential(*enc_layers)





