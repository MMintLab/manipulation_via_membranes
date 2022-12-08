import torch
import torch.nn as nn
import pytorch3d.transforms as batched_trs

from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule
from manipulation_via_membranes.bubble_learning.models.aux.img_encoder import ImageEncoder
from manipulation_via_membranes.bubble_learning.models.aux.img_decoder import ImageDecoder
from manipulation_via_membranes.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from manipulation_via_membranes.bubble_learning.models.dynamics_model_base import DynamicsModelBase
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis
from manipulation_via_membranes.bubble_learning.aux.pose_loss import ModelPoseLoss
from manipulation_via_membranes.bubble_learning.aux.visualization_utils.pose_visualization import get_object_pose_images_grid, get_angle_from_axis_angle


class ObjectPoseDynamicsModel(DynamicsModelBase):

    def __init__(self, *args, num_to_log=40, input_batch_norm=True, loss_name='pose_loss', **kwargs):
        self.num_to_log = num_to_log
        self.input_batch_norm = input_batch_norm
        super().__init__(*args, **kwargs)
        self.dyn_model = self._get_dyn_model()
        self.loss_name = loss_name
        self.pose_loss = ModelPoseLoss()
        self.plane_normal = nn.Parameter(torch.tensor([1, 0, 0], dtype=torch.float), requires_grad=False)
        sizes = self._get_sizes()
        self.dyn_input_batch_norm = nn.BatchNorm1d(
            num_features=sizes['dyn_input_size'])  # call eval() to freeze the mean and std estimation
        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'object_pose_dynamics_model'

    def _get_img_encoder(self):
        sizes = self._get_sizes()
        img_size = sizes['imprint']# (C_in, W_in, H_in)
        img_encoder = ImageEncoder(input_size=img_size,
                                   latent_size=self.img_embedding_size,
                                   num_convs=self.encoder_num_convs,
                                   conv_h_sizes=self.encoder_conv_hidden_sizes,
                                   ks=self.ks,
                                   num_fcs=self.num_encoder_fcs,
                                   fc_hidden_size=self.fc_h_dim,
                                   activation=self.activation)
        return img_encoder

    def _get_dyn_input_size(self, sizes):
        dyn_input_size = sizes['init_object_pose'] + sizes['init_pos'] + sizes['init_quat'] + self.object_embedding_size + sizes['action']
        return dyn_input_size

    def _get_dyn_output_size(self, sizes):
        dyn_output_size = sizes['init_object_pose']
        return dyn_output_size

    def forward(self, obj_pose, pos, ori, object_model, action):
        # sizes = self._get_sizes()
        # obj_pos_size = sizes['object_position']
        # obj_quat_size = sizes['object_orientation']
        # obj_pose_size = obj_pos_size + obj_quat_size
        obj_model_emb = self.object_embedding_module(object_model)  # (B, imprint_emb_size)
        dyn_input = torch.cat([obj_pose, pos, ori, obj_model_emb, action], dim=-1)
        if self.input_batch_norm:
            dyn_input = self.dyn_input_batch_norm(dyn_input)
        dyn_output = self.dyn_model(dyn_input)
        obj_pose_next = dyn_output # we only predict object_pose
        return (obj_pose_next,)

    def get_state_keys(self):
        state_keys = ['init_object_pose', 'init_pos', 'init_quat', 'object_model']
        return state_keys
    
    def get_input_keys(self):
        input_keys = ['init_object_pose', 'init_pos', 'init_quat', 'object_model']
        return input_keys

    def get_model_output_keys(self):
        output_keys = ['init_object_pose']
        return output_keys

    def get_next_state_map(self):
        next_state_map = {
            'init_object_pose': 'final_object_pose'
        }
        return next_state_map

    def _compute_loss(self, obj_pose_pred, obj_pose_gth, object_model):
        if self.loss_name == 'pose_loss':
            # MSE Loss on position and orientation (encoded as aixis-angle 3 values)
            axis_angle_pred = obj_pose_pred[..., 3:]
            R_pred = batched_trs.axis_angle_to_matrix(axis_angle_pred)
            t_pred = obj_pose_pred[..., :3]
            axis_angle_gth = obj_pose_gth[..., 3:]
            R_gth = batched_trs.axis_angle_to_matrix(axis_angle_gth)
            t_gth = obj_pose_gth[..., :3]
            # Transform object to be aligned with z axis in grasp frame
            frame_axis_angle = torch.tensor([0, -torch.pi/2, 0]).unsqueeze(0)
            frame_rotation = batched_trs.axis_angle_to_matrix(frame_axis_angle)
            frame_translation = torch.zeros(1, 3)
            object_model = self.pose_loss._transform_model_points(frame_rotation, frame_translation, object_model)
            #Compute loss
            loss = self.pose_loss(R_1=R_pred, t_1=t_pred, R_2=R_gth, t_2=t_gth, model_points=object_model)
        elif self.loss_name == 'mse':
            loss = self.mse_loss(obj_pose_pred, obj_pose_gth)
        else:
            raise NotImplementedError('Loss named {} not implemented yet.'.format(self.loss_name))

        return loss

    def _step(self, batch, batch_idx, phase='train'):
        action = batch['action']
        object_model = batch['object_model']

        model_input = self.get_model_input(batch)
        ground_truth = self.get_model_output(batch)
        model_output = self.forward(*model_input, action)
        loss = self._compute_loss(*model_output, *ground_truth, object_model)

        # Log the results: -------------------------
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # Log the images: -------------------------
        self._log_object_pose_images(obj_pose_pred=model_output[0][:self.num_to_log], obj_pose_gth=ground_truth[0][:self.num_to_log], phase=phase)
        return loss

    def _log_object_pose_images(self, obj_pose_pred, obj_pose_gth, phase):
        grid = get_object_pose_images_grid(obj_pose_pred, obj_pose_gth, self.plane_normal)
        self.logger.experiment.add_image('pose_estimation_{}'.format(phase), grid, self.global_step)

