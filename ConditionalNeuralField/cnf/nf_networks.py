import torch
from torch import nn
import numpy as np
import random
from einops import rearrange
import math

from ConditionalNeuralField.cnf.components import BatchLinear, FeatureMapping, NLS_AND_INITS, Activation
from ConditionalNeuralField.cnf.initialization import *


class Encoder(nn.Module):
    '''
    Encoder for each snapshot to get the latent vector
    based on: https://github.com/dc3042/CROM_offline_training/blob/main/cromnet.py#L470
    '''
    def __init__(self, data_format, lbllength, ks=6, strides=4, siren=True, omega_0=30):
        super().__init__()
        self.lbllength = lbllength
        self.siren = siren
        self.omega_0 = omega_0

        self.ks = ks
        self.strides = strides

        # goal: [o_dim, npoints] --> 64
        self.layers_conv = nn.ModuleList()
        l_in = data_format['npoints']
        layer_cnt = 0
        
        while True:
            l_out = self.conv1dLayer(l_in, self.ks, self.strides)
            if data_format['o_dim']*l_out >= lbllength*2:
                l_in = l_out
                layer_cnt += 1
                self.layers_conv.append(nn.Conv1d(data_format['o_dim'], data_format['o_dim'], self.ks, self.strides))
            else:
                break
        print(f"***Encoder has: {layer_cnt} conv layers")
        self.enc10 = nn.Linear(data_format['o_dim']*l_in, lbllength*2)
        self.enc11 = nn.Linear(lbllength*2, lbllength)
        
        self.act = Activation(self.siren, self.omega_0)
        self.init_weights()
        
        self.expand_dims = " ".join(["1" for _ in range(data_format['dims'])])
        self.expand_dims = f"N f -> N {self.expand_dims} f"
    
    def init_weights(self):
        with torch.no_grad():
            for m in self.children():
                
                random.seed(0)
                seed_number = random.randint(0, 100)
                random.seed(0)
                torch.manual_seed(0)
                
                if type(m) == nn.Linear:
                    if self.siren:
                        m.weight.uniform_(-np.sqrt(6 / m.in_features) / self.omega_0, 
                                            np.sqrt(6 / m.in_features) / self.omega_0)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                
                elif type(m) == nn.ModuleList:
                    for c in m:
                        if self.siren:
                            c.weight.uniform_(-np.sqrt(6 / self.ks) / self.omega_0, 
                                            np.sqrt(6 / self.ks) / self.omega_0)
                        else:
                            nn.init.xavier_uniform_(c.weight)
                
                torch.manual_seed(torch.initial_seed())

            if self.siren:
                self.layers_conv[0].weight.uniform_(-1 / self.ks, 1 / self.ks)
    
    def forward(self, state):
        if len(state.shape) <= 2:
            raise ValueError(f"Expected input dimension >= 3, but got {state.dim()}")
        elif len(state.shape) == 4:
            state = rearrange(state, 'b h w c -> b (h w) c')
        elif len(state.shape) == 5:
            state = rearrange(state, 'b x y z c -> b (x y z) c')
        else:
            raise NotImplementedError(f"Input dimension {state.dim()} not supported")
        assert state.dim() == 3, f"Expected input dimension 3, but got {state.dim()}"
        state = torch.transpose(state, 1, 2)    # <b, npoints, o_dim> --> <b, o_dim, npoints>
        
        for layer in self.layers_conv:
            state = self.act(layer(state))
        
        state = torch.transpose(state, 1, 2)

        state = state.reshape(-1, state.size(1)*state.size(2))
        state = self.act(self.enc10(state))
        xhat = self.act(self.enc11(state))    # <b, lbllength>
        
        out = rearrange(xhat, self.expand_dims)

        return out

    def conv1dLayer(self, l_in, ks, strides):
        # calculate the output length of the conv1d layer
        return math.floor(float(l_in -(ks-1)-1)/strides + 1)



# siren auto decoder: FILM

class SIRENAutodecoder_film(nn.Module):
    '''
    siren network with auto decoding 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
                 premap_mode = None, omega_0=30, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if self.premap_mode is not None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features     
        
        # self.fourier_transform = GaussianFourierFeatureTransform(in_coord_features, int(hidden_features//2), 10)
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]

        if nonlinearity == 'sine':
            self.nl.w0 = omega_0
        
        self.weight_init = nl_weight_init

        self.nf_net = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features,out_features)])
        self.hb_net = nn.ModuleList([BatchLinear(in_latent_features,hidden_features,bias = False) for i in range(num_hidden_layers+1)])

        if self.weight_init is not None:
            self.nf_net.apply(self.weight_init(omega_0))
            self.hb_net.apply(self.weight_init(omega_0))

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.nf_net[0].apply(first_layer_init)
            self.hb_net[0].apply(first_layer_init)

        if bias_init is not None:
            self.hb_net.apply(bias_init)

    def forward(self, coords, latents):
        # coords: <b,h,w,c_coord>
        # latents: <b,h,w,c_latent>  h, w are broadcasted

        # premap 
        if self.premap_mode is not None: 
            x = self.premap_layer(coords)
        else: 
            x = coords
        
        # fourier transform
        # x = self.fourier_transform(coords)
        
        # pass it through  the nf network 
        for i in range(len(self.nf_net) -1):
            x = self.nf_net[i](x) + self.hb_net[i](latents)
            x = self.nl(x)
        x = self.nf_net[-1](x)
        
        return x 


    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False


# siren auto decoder: modified FILM
class SIRENAutodecoder_mdf_film(nn.Module):
    '''
    siren network with auto decoding 
    mdf mean the conditioning is full projection 
    '''
    def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
                 premap_mode = None, omega_0=30, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode ==None: 
            self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim # update the nf in features  
        
        # self.fourier_transform = GaussianFourierFeatureTransform(in_coord_features, int(hidden_features//2), 10)   

        self.first_layer_init = None
                        
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
        
        if nonlinearity == 'sine':
            self.nl.w0 = omega_0

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init # those are default init funcs 

        # create the net for the nf 
        self.nf_net = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
                                  [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_layers)] + 
                                  [BatchLinear(hidden_features,out_features)])

        # create the net for the weights and bias, the hypernet it self has no bias. 
        self.hw_net = nn.ModuleList([BatchLinear(in_latent_features,in_coord_features*hidden_features,bias = False)]+
                                    [BatchLinear(in_latent_features,hidden_features*hidden_features,bias = False) for i in range(num_hidden_layers)])
                                    # [BatchLinear(in_latent_features,hidden_features*out_features,bias = False)])
        self.hb_net = nn.ModuleList([BatchLinear(in_latent_features,hidden_features,bias = False) for i in range(num_hidden_layers+1)])
                                    #  [BatchLinear(in_latent_features,out_features*out_features,bias = False)])

        if self.weight_init is not None:
            self.nf_net.apply(self.weight_init(omega_0))

        # if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
        #     self.nf_net[0].apply(first_layer_init)
        self.init_hyper_layer()     

    def init_hyper_layer(self):
        # init weights
        self.hw_net.apply(init_weights_uniform_siren_scale)
        self.hb_net.apply(init_weights_uniform_siren_scale)

    def forward(self, coords, latents):
        # _, h_size, w_size, coord_size = coords.shape
        
        # while len(coords.shape) < 4:
        #     coords = coords[:, None, ...]
        #     latents = latents[:, None, ...]
        # print("coords shape:", coords.shape, "latents shape:", latents.shape)
        t_size = latents.shape[0]
        if len(coords.shape) == 4:  # (H, W)
            dim = 2
        elif len(coords.shape) == 3:  # 3D case with flattened coordinates
            dim = 1
        else:
            raise ValueError(f"Expected input coord dimension 3 or 4, but got {coords.dim()}")
        # coords: <t,h,w,c_coord> or <1, h,w,c_coords>
        # latents: <t,h,w,c_latent> or <t, 1,1, c>
        
        # premap
        if not self.premap_mode ==None: 
            x = self.premap_layer(coords)
        else: 
            x = coords
        
        # fourier transform
        # x = self.fourier_transform(coords)

        # pass it through  the nf network 
        for i in range(len(self.nf_net) -1):
            
            reshape_dims = (t_size,) + (1,) * dim + self.nf_net[i].weight.shape

            x = (
                    self.nf_net[i](x) + 
                    torch.einsum(
                        '...i,...ji->...j', 
                        x, 
                        # self.hw_net[i](latents).reshape((t_size, 1, 1)+self.nf_net[i].weight.shape)
                        self.hw_net[i](latents).reshape(reshape_dims)
                    )
                    + self.hb_net[i](latents)
                )
            
            x = self.nl(x)

        x = self.nf_net[-1](x)

        return x 
    
    def disable_gradient(self):
        for param in self.parameters():
            param.requires_grad = False



# class SIRENAutodecoder_mdf_film(nn.Module):
#     '''
#     siren network with auto decoding 
#     mdf mean the conditioning is full projection 
#     '''
#     def __init__(self, in_coord_features, in_latent_features, out_features, num_hidden_layers, hidden_features,
#                  outermost_linear=False, nonlinearity='sine', weight_init=None,bias_init=None,
#                  premap_mode = None,dim=1,omega_0=30.0, **kwargs):
#         super().__init__()
#         self.premap_mode = premap_mode
#         if not self.premap_mode ==None: 
#             self.premap_layer = FeatureMapping(in_coord_features,mode = premap_mode, **kwargs)
#             in_coord_features = self.premap_layer.dim # update the nf in features     
#             print("applying feature mapping")

#         self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
#         if nonlinearity == 'sine':
#             self.nl.w0 = omega_0

#         self.weight_init = nl_weight_init

#         # create the net for the nf 
#         self.nf_net = nn.ModuleList([BatchLinear(in_coord_features,hidden_features)] + 
#                                   [BatchLinear(hidden_features,hidden_features) for i in range(num_hidden_layers)] + 
#                                   [BatchLinear(hidden_features,out_features)])

#         # create the net for the weights and bias, the hypernet it self has no bias. 
#         self.hw_net = nn.ModuleList([BatchLinear(in_latent_features,in_coord_features*hidden_features,bias = False)]+
#                                     [BatchLinear(in_latent_features,hidden_features*hidden_features,bias = False) for i in range(num_hidden_layers)])
#                                     # [BatchLinear(in_latent_features,hidden_features*out_features,bias = False)])
#         self.hb_net = nn.ModuleList([BatchLinear(in_latent_features,hidden_features,bias = False) for i in range(num_hidden_layers+1)])
#                                     #  [BatchLinear(in_latent_features,out_features*out_features,bias = False)])
#         self.dim = dim

#         if self.weight_init is not None:
#             print("applying weight init")
#             self.nf_net.apply(self.weight_init(omega_0))
#             self.hw_net.apply(self.weight_init(omega_0))
#             self.hb_net.apply(self.weight_init(omega_0))
        
#         if first_layer_init is not None:
#             print("applying first layer init")
#             self.nf_net[0].apply(first_layer_init)
#             self.hw_net[0].apply(first_layer_init)
#             self.hb_net[0].apply(first_layer_init)

#     def forward(self, coords, latents):
#         # _, h_size, w_size, coord_size = coords.shape
#         t_size = latents.shape[0]
#         # coords: <t,h,w,c_coord> or <1, h,w,c_coords>
#         # latents: <t,h,w,c_latent> or <t, 1,1, coords>

#         # premap 
#         if not self.premap_mode ==None: 
#             x = self.premap_layer(coords)
#         else: 
#             x = coords
#         # pass it through  the nf network 
#         for i in range(len(self.nf_net) -1):
#             # print("x shape:", x.shape)
#             reshape_dims = (t_size,) + (1,) * self.dim + self.nf_net[i].weight.shape
#             x = (
#                 self.nf_net[i](x) +
#                 torch.einsum(
#                     '...i,...ji->...j',
#                     x,
#                     self.hw_net[i](latents).reshape(reshape_dims)
#                 )
#                 + self.hb_net[i](latents)
#             )
            
#             x = self.nl(x)

#         x = self.nf_net[-1](x)

#         return x