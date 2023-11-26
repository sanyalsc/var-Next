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
        super(varNext, self).__init__()
        if type(cfg_file)!= dict:
            with open(cfg_file,'r') as f:
                cfg_file = json.load(f)
        layers = cfg_file['layers']
        # Build encoder
        enc_layers = []
        enc_shapes = [cfg_file['input_shape']]
        for layer in layers['encoder']:
            conv = layer['conv']
            activation = nn.LeakyReLU()  
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(conv['in_channel'],conv['out_channel'],conv['kernel_size'],**conv['kwargs']),
                    nn.BatchNorm2d(conv['out_channel']),
                    activation,
                    torch.nn.Dropout(p=0.1)
            )
            )
            enc_shapes.append(
                conv_output_shape(enc_shapes[-1],
                                  kernel_size = conv['kernel_size'],
                                  stride = conv['kwargs']['stride'])
                                )
        
        #bottleneck
        unflatten_dim = [layers['encoder'][-1]['conv']['out_channel'],enc_shapes[-1][0],enc_shapes[-1][1]]
        self.pack_LL = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor(unflatten_dim)),out_features=512),
            nn.LeakyReLU()
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
        self.d1 = nn.Sequential(
            nn.Linear(layers['latent'],torch.prod(torch.tensor(unflatten_dim))),
            nn.LeakyReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=unflatten_dim)

        for layer in layers['decoder']:
            conv = layer['conv']
            activation = nn.LeakyReLU()
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(conv['in_channel'],conv['out_channel'],conv['kernel_size'],**conv['kwargs']),
                    nn.BatchNorm2d(conv['out_channel']),
                    activation,
                    torch.nn.Dropout(p=0.1)
            )
            )
        
        self.decoder = nn.Sequential(*dec_layers)


    def encode(self,x):
        x = self.encoder(x)
        print(f'unflat enc: {x.shape}')
        x = torch.flatten(x,start_dim=1)
        x = self.pack_LL(x)
        mu = self.mu(x)
        sigma = torch.exp(self.sig(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def decode(self,x):
        x = self.d1(x)
        x = self.unflatten(x)
        print(f' dec inp {x.shape}')
        return self.decoder(x)

    def forward(self,x):
        x = self.encode(x)
        return self.decode(x)
    

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w