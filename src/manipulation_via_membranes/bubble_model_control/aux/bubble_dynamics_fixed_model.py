import torch
import torch.nn as nn


class BubbleDynamicsFixedModel(nn.Module):

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_fixed_model'

    @property
    def name(self):
        return self.get_name()

    def forward(self, imprint, wrench, pos, ori, object_model, action):
        imprint_next = imprint # Assume that the object will stay at the same place and therefore the imprint will be the same
        wrench_next = wrench
        return imprint_next, wrench_next

    def get_state_keys(self):
        state_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return state_keys

    def get_input_keys(self):
        input_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return input_keys

    def get_model_output_keys(self):
        output_keys = ['init_imprint', 'init_wrench']
        return output_keys

    def get_next_state_map(self):
        next_state_map = {
            'init_imprint': 'final_imprint',
            'init_wrench': 'final_wrench',
        }
        return next_state_map

