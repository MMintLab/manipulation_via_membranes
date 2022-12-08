import torch

from manipulation_via_membranes.bubble_learning.models.bubble_dynamics_model_base import BubbleDynamicsModelBase


class BubbleEnd2EndDynamicsModel(BubbleDynamicsModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'bubble_end2end_dynamics_model'

    def _get_dyn_input_size(self, sizes):
        dyn_input_size = self.img_embedding_size + sizes['init_wrench'] + sizes['init_pos'] + sizes['init_quat'] + self.object_embedding_size + sizes['action']
        return dyn_input_size

    def _get_dyn_output_size(self, sizes):
        dyn_output_size = self.img_embedding_size + sizes['init_wrench'] + sizes['init_object_pose']
        return dyn_output_size

    def forward(self, imprint, wrench, pos, ori, object_model, action):
        sizes = self._get_sizes()
        imprint_input_emb = self.autoencoder.encode(imprint)
        obj_model_emb = self.object_embedding_module(object_model)  # (B, imprint_emb_size)
        dyn_input = torch.cat([imprint_input_emb, wrench, pos, ori, obj_model_emb, action], dim=-1)
        dyn_output = self.dyn_model(dyn_input)
        output_sizes = [self.img_embedding_size, sizes['init_wrench'], sizes['init_object_pose']]
        imprint_emb_next, wrench_next, obj_pose_next  = torch.split(dyn_output, output_sizes, dim=-1)
        imprint_next = self.autoencoder.decode(imprint_emb_next)        
        return imprint_next, wrench_next, obj_pose_next

    def get_state_keys(self):
        state_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model', 'init_object_pose']
        return state_keys
    
    def get_input_keys(self):
        input_keys = ['init_imprint', 'init_wrench', 'init_pos', 'init_quat',
                      'object_model']
        return input_keys

    def get_model_output_keys(self):
        output_keys = ['init_imprint', 'init_wrench', 'init_object_pose']
        return output_keys

    def get_next_state_map(self):
        next_state_map = {
            'init_imprint': 'final_imprint',
            'init_wrench': 'final_wrench',
            'init_object_pose': 'final_object_pose'
        }
        return next_state_map

    def _compute_loss(self, imprint_pred, wrench_pred, obj_pose_pred, imprint_gth, wrench_gth, obj_pose_gth):
        # MSE Loss on position and orientation (encoded as axis-angle 3 values)
        pose_loss = self.mse_loss(obj_pose_pred, obj_pose_gth)
        imprint_loss = self.mse_loss(imprint_pred, imprint_gth)
        wrench_loss = self.mse_loss(wrench_pred, wrench_gth)
        loss = imprint_loss + wrench_loss + pose_loss # TODO: Consider adding different weights
        return loss