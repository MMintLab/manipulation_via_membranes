import numpy as np
import torch
import torch.nn as nn
import os
import sys

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/ContactInvariances')[0], 'ContactInvariances')
package_path = os.path.join(project_path, 'contact_invariances', 'learning')
sys.path.append(project_path)

from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule


class ImageDecoder(nn.Module):
    """
    Module composed by FC layers and 2D Inverse Convolutions (Transposed Conv)
    """
    def __init__(self, output_size, latent_size, num_convs=3, conv_h_sizes=None, ks=4, stride=1, padding=0, dilation=1, num_fcs=2, pooling_size=1, fc_hidden_size=50, activation='relu'):
        super().__init__()
        self.output_size = output_size # (C_out, W_out, H_out)
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
        self.conv_decoder, self.conv_in_size, self.img_conv_sizes = self._get_conv_decoder()
        self.fc_decoder = self._get_fc_decoder()

    def forward(self, z):
        batch_size = z.size(0) # shape (Batch_size, ..., latent_size)
        conv_in = self.fc_decoder(z) # adjust the shape for the convolutions
        conv_in = conv_in.view(z.size()[:-1] + tuple(self.conv_in_size))  # shape (Batch_size, ..., C_in, H_in, W_in)
        conv_out = self.conv_decoder(conv_in) # shape (Batch_size, ..., C_out, H_out, W_out)
        return conv_out

    def _get_convs_h_sizes(self, num_convs, conv_h_sizes):
        if conv_h_sizes is None:
            hidden_dims = [self.output_size[0]]*num_convs + [self.output_size[0]]
        else:
            hidden_dims = conv_h_sizes + [self.output_size[0]]
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

    def _get_conv_decoder(self):
        conv_modules = []
        sizes = [self.output_size[1:]]
        for i in reversed(range(len(self.hidden_dims)-1)):
            h_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i+1]
            ks = self.ks[i]
            stride = self.stride[i]
            padding = self.padding[i]
            dilation = self.dilation[i]
            pooling_size_i = self.pooling_size[i]
            _size_up = (sizes[-1] - 1 + 2 * padding - dilation * (ks - 1))
            in_size_i = np.floor(_size_up / stride + 1).astype(np.int64)
            expected_out_size = (in_size_i - 1)*stride - 2*padding + dilation*(ks - 1) + 1
            out_padding = sizes[-1] - expected_out_size
            conv_i = nn.ConvTranspose2d(in_channels=h_dim, out_channels=out_dim, kernel_size=ks, padding=padding, dilation=dilation, stride=stride, output_padding=out_padding)
            conv_modules.append(conv_i)
            sizes.append(in_size_i)
            if i < len(self.hidden_dims)-2:
                conv_modules.append(self.act)
        conv_modules.reverse()
        if len(conv_modules) > 0:
            conv_encoder = nn.Sequential(*conv_modules)
        else:
            conv_encoder = nn.Identity() # no operation needed since there are no convolutions
        # compute the tensor sizes:
        sizes.reverse()
        conv_img_in_size_wh = sizes[0]
        conv_img_in_size = np.insert(conv_img_in_size_wh, 0, self.hidden_dims[0]) # ( C_in, H_in, W_in)
        return conv_encoder, conv_img_in_size, sizes

    def _get_fc_decoder(self):
        fc_out_size = int(np.prod(self.conv_in_size))
        sizes = [self.latent_size] + [self.fc_hidden_size]*(self.num_fcs-1) + [fc_out_size]
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
