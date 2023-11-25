####################
# Shantanu Sanyal
# sanyalster@gmail.com
#
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
        layers = cfg_file['layers']
        # Build encoder
        enc_layers = []
        enc_shapes = [cfg_file['input_shape']]
        for layer in layers['encoder']:
            conv = layer['conv']
            if layer['activation']=='leakyrelu':
                activation = nn.LeakyReLU()
            else:
                activation = nn.ReLU()  
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(conv['in_channels'],conv['out_channels'],conv['kernel_size'],**conv),
                    nn.BatchNorm2d(conv['out_channels']),
                    activation,
                    torch.nn.Dropout(p=0.1)
            )
            enc_shapes.append(
                conv_output_shape(enc_shapes[-1],
                                  kernel_size = conv['kernel_size'],
                                  stride = conv['stride'])
                                )
        
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(layers['mu'],layers['latent'])
        self.sig = nn.Linear(layers['var'],layers['latent'])
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        
        # Build decoder
        dec_layers = []
        unflatten_dim = [enc_shapes[-1][0],enc_shapes[-1][1],layers['encoder'][-1]['out_channels']]
        self.d1 = nn.Linear(layers['latent'],torch.prod(layers['unflatten']))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=unflatten_dim)

        for layer in layers['decoder']:
            conv = layer['conv']
            activation = nn.LeakyReLU()
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2D(conv['in_channels'],conv['out_channels'],conv['kernel_size'],**conv),
                    nn.BatchNorm2d(conv['out_channels']),
                    activation,
                    torch.nn.Dropout(p=0.1)
            )
        
        self.decoder = nn.Sequential(*enc_layers)


    def encode(x):
        x = self.encoder(x)
        mu = self.mu(x)
        sigma = torch.exp(self.sig(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def decode(x):
        x = self.d1(x)
        x = self.unflatten(x)
        return self.decoder(x)

    def forward(x):
        self.encode(x)
        return self.decode(x)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w