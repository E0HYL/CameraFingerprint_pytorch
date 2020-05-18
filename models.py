#!/usr/bin/env python
# coding: utf-8

"""
Definition of the FFDNet model and its custom layers

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import functions
import os

class UpSampleFeatures(nn.Module):
    r"""Implements the last layer of FFDNet
    """
    def __init__(self):
        super(UpSampleFeatures, self).__init__()
    def forward(self, x):
        return functions.upsamplefeatures(x)

class IntermediateDnCNN(nn.Module):
    r"""Implements the middel part of the FFDNet architecture, which
    is basically a DnCNN net
    """
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateDnCNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        if self.input_features == 5:
            self.output_features = 4 #Grayscale image
        elif self.input_features == 4:
            self.output_features = 3 #RGB image
        else:
            raise Exception('Invalid number of input features')

        device = 'cuda:0'
        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_features,\
                                out_channels=self.middle_features,\
                                kernel_size=self.kernel_size,\
                                padding=self.padding,\
                                bias=True).to(device))
        layers.append(nn.ReLU(inplace=True).to(device))
        for i in range(self.num_conv_layers-2):
            if i == 4:
                device = 'cuda:1'
            elif i == 9:
                device = 'cuda:2'
            elif i == 14:
                device = 'cuda:3'
            layers.append(nn.Conv2d(in_channels=self.middle_features,\
                                    out_channels=self.middle_features,\
                                    kernel_size=self.kernel_size,\
                                    padding=self.padding,\
                                    bias=True).to(device))
            # layers.append(nn.BatchNorm2d(self.middle_features).to(device))
            layers.append(nn.ReLU(inplace=True).to(device))
        layers.append(nn.Conv2d(in_channels=self.middle_features,\
                                out_channels=self.output_features,\
                                kernel_size=self.kernel_size,\
                                padding=self.padding,\
                                bias=False).to(device))
        self.itermediate_dncnn = nn.Sequential(*layers)
        # print(self.itermediate_dncnn)
    def forward(self, x):
        device = 'cuda:0'
        out = x
        with torch.no_grad():
            for i, layer in enumerate(self.itermediate_dncnn):
                if i == 10:
                    device = 'cuda:1'
                elif i == 20:
                    device = 'cuda:2'
                elif i == 30:
                    device = 'cuda:3'
                #print(layer.parameters[0])
                out = layer(out.to(device))
                # print(out.permute(2,3,1,0))
                # input()
        #out = self.itermediate_dncnn(x)
        return out

class FFDNet(nn.Module):
    r"""Implements the FFDNet architecture
    """
    def __init__(self, num_input_channels):
        super(FFDNet, self).__init__()
        self.num_input_channels = num_input_channels
        if self.num_input_channels == 1:
            # Grayscale image
            self.num_feature_maps = 64
            self.num_conv_layers = 15
            self.downsampled_channels = 5
            self.output_features = 4
        elif self.num_input_channels == 3:
            # RGB image
            self.num_feature_maps = 64
            self.num_conv_layers = 20
            self.downsampled_channels = 4
            self.output_features = 3
        else:
            raise Exception('Invalid number of input features')

        self.intermediate_dncnn = IntermediateDnCNN(\
                input_features=self.downsampled_channels,\
                middle_features=self.num_feature_maps,\
                num_conv_layers=self.num_conv_layers)
        self.upsamplefeatures = UpSampleFeatures()

    def forward(self, x, noise_sigma):
        concat_noise_x = functions.concatenate_input_noise_map(\
                x.data, noise_sigma.data)
        concat_noise_x = Variable(concat_noise_x)
        # print(concat_noise_x.permute(2,3,1,0))
        # input()
        pred_noise = self.intermediate_dncnn(concat_noise_x)
        # h_dncnn = self.intermediate_dncnn(concat_noise_x)
        # pred_noise = self.upsamplefeatures(h_dncnn)
        return pred_noise
