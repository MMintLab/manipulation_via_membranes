import torch
import torch.nn as nn


class FCModule(nn.Module):

    def __init__(self, sizes, skip_layers=None, activation='relu'):
        self.sizes = sizes
        self.skip_layers = skip_layers
        super().__init__()
        self.act = self._get_activation(activation)
        self.layers = self._get_layers()

    def _get_layers(self):
        layers = []
        prev_skip_size = self.sizes[0] # input size
        for i, s_i in enumerate(self.sizes[:-1]):
            in_size_i = s_i
            if self.skip_layers is not None:
                if (i + 1) % (self.skip_layers + 1) == 0:
                    in_size_i += prev_skip_size
                    prev_skip_size = self.sizes[i + 1]
            l_i = nn.Linear(in_size_i, self.sizes[i + 1])
            layers.append(l_i)
            # if i < len(sizes) - 2:
            #     layers.append(self.act)
        layers = nn.ModuleList(layers)
        return layers

    def forward(self, x):
        x_in = torch.clone(x)
        x_res = torch.clone(x_in)  # stores the output from the previous skip layer
        for i, layer in enumerate(self.layers):
            if self.skip_layers is not None:
                if (i + 1) % (self.skip_layers + 1) == 0:
                    x = torch.cat([x, x_res], dim=-1)
            x = layer(x)
            if self.skip_layers is not None:
                if (i + 1) % (self.skip_layers + 1) == 0:
                    x_res = torch.clone(x)  # store the result to input in the next skip layer
            if i < len(self.layers) - 1:
                x = self.act(x)
        x_out = x
        return x_out

    @classmethod
    def _get_activation(cls, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation is None:
            return nn.Identity() # no activation
        else:
            raise NotImplementedError('Activation {} not supported'.format(activation))