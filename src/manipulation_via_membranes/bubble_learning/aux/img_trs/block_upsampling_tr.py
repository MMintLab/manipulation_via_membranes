import numpy as np
import torch
import torch.nn.functional as F
import copy
import abc


class BlockUpSamplingTr(abc.ABC):
    def __init__(self, factor_x, factor_y, method='repeat', keys_to_tr=None):
        super().__init__()
        self.factor_x = factor_x
        self.factor_y = factor_y
        self.method = method
        self.keys_to_tr = keys_to_tr

    def __call__(self, sample):
        if self.keys_to_tr is None:
            # transform all that has quat in the key
            old_keys = copy.deepcopy(list(sample.keys()))
            for k in old_keys:
                v = sample[k]
                if 'imprint' in k:
                    sample['{}_upsampled'.format(k)] = v  # store the unsampled one
                    sample[k] = self._tr(v)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    v = sample[key]
                    sample['{}_upsampled'.format(key)] = v  # store the unsample one
                    sample[key] = self._tr(sample[key])
        return sample

    def inverse(self, sample):
        # apply the inverse transformation
        if self.keys_to_tr is None:
            # trasform all that has quat in the key
            for k, v in sample.items():
                if 'imprint' in k:
                    sample[k] = sample['{}_upsampled'.format(k)] # restore the original
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample[key] = sample['{}_upsampled'.format(key)] # restore the original
        return sample

    def _tr(self, x):
        # ---- repeat upsampling ----
        is_tensor = type(x) is torch.Tensor
        if self.method == 'repeat':
            if is_tensor:
                x_upsampled = x.repeat_interleave(self.factor_x, dim=-2).repeat_interleave(self.factor_y, dim=-1)
            else:
                # numpy case
                x_upsampled = x.repeat(self.factor_x, axis=-2).repeat(self.factor_y, axis=-1)
        elif self.method in ['bilinear', 'bicubic']:
            # ---- interpolation upsampling -- (TODO)
            # Use pytorch interpolate for batched interpolation
            size_x = x.shape[-2]
            size_y = x.shape[-1]
            if is_tensor:
                x_t = x
            else:
                x_t = torch.tensor(x)
            x_t = x_t.reshape(-1, 1, size_x, x.shape[-1]) # add num channels (expected (batch, num_channels, depth, height))
            new_size = (size_x*self.factor_x, size_y*self.factor_y)
            x_upsampled_t = F.interpolate(x_t, size=new_size, mode=self.method, align_corners=True)
            x_upsampled = x_upsampled_t.reshape(*x.shape[:-2], *new_size)
            if not is_tensor:
                x_upsampled = x_upsampled.cpu().detach().numpy() # convert it back to numpy
        else:
            raise NotImplemented('method {} not available yet. Available methods: {}'.format(self.metod, ['repeat']))
        return x_upsampled

