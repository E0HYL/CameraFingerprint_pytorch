#!/usr/bin/env python
# coding: utf-8


import collections
from scipy.io import loadmat
import torch

d=collections.OrderedDict()
mat_model = loadmat('./models/FDnCNN_color.mat', squeeze_me=True, struct_as_record=False)
mat_model_layers = mat_model['net'].layers

n = 0
for mat_layer in mat_model_layers:
    layer_type = mat_layer.type
    layer_name = mat_layer.name
    print(layer_type, n)
    if layer_type == 'conv':
        mat_weights = mat_layer.weights[0]
        mat_bias = mat_layer.weights[1]
        key1 = 'intermediate_dncnn.itermediate_dncnn.%d.weight'%n
        d[key1] = torch.from_numpy(mat_weights.transpose([3,2,0,1]))
        if mat_bias.all():
            key2 = 'intermediate_dncnn.itermediate_dncnn.%d.bias'%n
            d[key2] = torch.from_numpy(mat_bias)
        n += 1
    elif layer_type == 'concat':
        pass
    elif layer_type == 'relu':
        n += 1
    else:
        raise TypeError('error')
        
torch.save(d, 'net_rgb.pth')
