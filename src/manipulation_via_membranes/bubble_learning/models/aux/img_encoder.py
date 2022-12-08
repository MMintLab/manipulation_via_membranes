import numpy as np
import torch
import torch.nn as nn
import os
import sys

from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule


class ImageEncoder(nn.Module):
    """
    Module composed by 2D Convolutions followed by FC layers
    """
    def __init__(self, input_size, latent_size, num_convs=3, conv_h_sizes=None, ks=4, stride=1, padding=0, dilation=1, num_fcs=2, pooling_size=1, fc_hidden_size=50, activation='relu'):
        super().__init__()
        self.input_size = input_size # (C_in, W_in, H_in)
        self.latent_size = latent_size
        self.num_convs, self.hidden_dims = self._get_convs_h_sizes(num_convs, conv_h_sizes)
        self.ks = self._get_conv_property(ks)
        self.stride = self._get_conv_property(stride)
        self.padding = self._get_conv_property(padding)
        self.dilation = self._get_conv_property(dilation)
        self.pooling_size = self._get_conv_property(pooling_size)
        self.num_fcs = num_fcs
        self.fc_hidden_size = fc_hidden_size
        self.act = self._get_activation(activation) # only used in conv
        self.conv_encoder, self.conv_out_size, self.img_conv_sizes = self._get_conv_encoder()
        self.fc_encoder = self._get_fc_encoder()

    def forward(self, x):
        batch_size = x.size(0) # shape (Batch_size, ..., C_in, H_in, W_in)
        conv_out = self.conv_encoder(x) # shape (Batch_size, ..., C_out, H_out, W_out)
        fc_in = torch.flatten(conv_out, start_dim=-3) # flatten the last 3 dims
        z = self.fc_encoder(fc_in)
        return z

    def _get_convs_h_sizes(self, num_convs, conv_h_sizes):
        if conv_h_sizes is None:
            hidden_dims = [self.input_size[0]] + [self.input_size[0]*3]*num_convs
        else:
            hidden_dims = [self.input_size[0]] + conv_h_sizes
            num_convs = len(conv_h_sizes)
        return num_convs, hidden_dims

    def _get_conv_property(self, ks):
        if type(ks) in [int]:
            # single ks, we need to extend it ot num_convs
            ks = np.array([ks]*self.num_convs)
        elif type(ks) in [np.ndarray, torch.Tensor]:
            pass
        elif type(ks) in [list, tuple]:
            ks = np.asarray(ks)
        else:
            raise NotImplementedError(f'Option to set the conv property with {ks} is not available yet. Please, check the requirements at img_encoder.py')
        assert len(
            ks) == self.num_convs, f'We must have the same args as num_convs. len(arg)={len(ks)}, num_convs={self.num_convs}'
        return ks

    def _get_conv_encoder(self):
        conv_modules = []
        sizes = [self.input_size[1:]]
        for i, h_dim in enumerate(self.hidden_dims[:-1]):
            ks = self.ks[i]
            stride = self.stride[i]
            padding = self.padding[i]
            dilation = self.dilation[i]
            out_dim = self.hidden_dims[i + 1]
            pooling_size_i = self.pooling_size[i]
            conv_i = nn.Conv2d(in_channels=h_dim, out_channels=out_dim, kernel_size=ks, stride=stride, padding=padding, dilation=dilation)
            pooling_i = nn.AvgPool2d(kernel_size=pooling_size_i)
            conv_modules.append(conv_i)
            conv_modules.append(self.act)
            conv_modules.append(pooling_i)
            size_i = (np.floor((sizes[-1]+2*padding - dilation*(ks-1) - 1)/stride) + 1).astype(np.int64)
            size_i = np.floor_divide(size_i, pooling_size_i).astype(np.int64)
            sizes.append(size_i)
        conv_encoder = nn.Sequential(*conv_modules)
        conv_img_out_size_wh = sizes[-1]
        conv_img_out_size = np.insert(conv_img_out_size_wh, 0, self.hidden_dims[-1])
        return conv_encoder, conv_img_out_size, sizes

    def _get_fc_encoder(self):
        fc_in_size = np.prod(self.conv_out_size)
        sizes = [fc_in_size] + [self.fc_hidden_size]*(self.num_fcs-1) + [self.latent_size]
        fc_encoder = FCModule(sizes, activation='relu')
        return fc_encoder

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError('Activation {} not supported'.format(activation))