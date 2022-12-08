import torch.nn as nn

from manipulation_via_membranes.bubble_learning.models.bubble_dynamics_model import BubbleDynamicsModel


class BubbleLinearDynamicsModel(BubbleDynamicsModel):
    @classmethod
    def get_name(cls):
        return 'bubble_linear_dynamics_model'

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        dyn_input_size = sizes['dyn_input_size']
        dyn_output_size = sizes['dyn_output_size']
        dyn_model = nn.Linear(in_features=dyn_input_size, out_features=dyn_output_size, bias=False)
        return dyn_model


