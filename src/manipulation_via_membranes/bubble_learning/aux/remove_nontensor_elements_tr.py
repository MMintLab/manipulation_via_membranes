import torch


class RemoveNonTensorElementsTr(object):
    def __call__(self, sample):
        all_keys = list(sample.keys())
        all_types = [type(v) for v in sample.values()]
        for i, type_i in enumerate(all_types):
            key_i = all_keys[i]
            if not type_i in [torch.Tensor]:
                sample.pop(key_i)
        return sample

    def inverse(self, sample):
        # we cannot invert the transformation
        return sample
