import numpy as np
import os
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import pytorch_lightning as pl
import abc
import torchvision
import pytorch3d.transforms as batched_trs
import tf.transformations as tr
from matplotlib import cm


from manipulation_via_membranes.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule
from manipulation_via_membranes.aux.load_confs import load_object_models
from manipulation_via_membranes.bubble_learning.aux.pose_loss import PoseLoss
from manipulation_via_membranes.bubble_learning.aux.visualization_utils.image_grid import get_imprint_grid, get_batched_image_grid
from manipulation_via_membranes.bubble_learning.aux.visualization_utils.pose_visualization import get_object_pose_images_grid


class ICPApproximationModel(pl.LightningModule):

    def __init__(self, input_sizes, num_fcs=2, fc_h_dim=100,
                 skip_layers=None, lr=1e-4, dataset_params=None, activation='relu', load_autoencoder_version=0, object_name='marker', num_to_log=40, autoencoder_augmentation=False, loss_name='pose_loss'):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation
        self.object_name = object_name
        self.autoencoder_augmentation = autoencoder_augmentation
        self.object_model = self._get_object_model()
        self.loss_name = loss_name
        self.mse_loss = nn.MSELoss()
        self.pose_loss = PoseLoss(self.object_model)
        self.plane_normal = nn.Parameter(torch.tensor([1, 0, 0], dtype=torch.float), requires_grad=False)
        self.num_to_log = num_to_log
        self.autoencoder = self._load_autoencoder(load_version=load_autoencoder_version,
                                                  data_path=self.dataset_params['data_name'])
        self.autoencoder.freeze()
        self.img_embedding_size = self.autoencoder.img_embedding_size  # load it from the autoencoder
        self.pose_estimation_network = self._get_pose_estimation_network()

        self.save_hyperparameters()  # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'icp_approximation_model'

    @property
    def name(self):
        return self.get_name()

    def _get_pose_estimation_network(self):
        input_size = self.img_embedding_size
        output_size = self.input_sizes['object_pose']
        pen_sizes = [input_size] + [self.fc_h_dim]*self.num_fcs + [output_size]
        pen = FCModule(sizes=pen_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return pen

    def _get_object_model(self):
        model_pcs = load_object_models()
        object_model_ar = np.asarray(model_pcs[self.object_name].points)
        # Transform object to be aligned with z axis in grasp frame
        tr_matrix = tr.quaternion_matrix(tr.quaternion_from_euler(0, -np.pi / 2, 0))
        object_model_H = np.concatenate([object_model_ar, np.ones(object_model_ar.shape[:-1]+(1,))], axis=-1)
        object_model_tr = np.einsum('ij,kj->ki', tr_matrix, object_model_H)
        object_model = object_model_tr[..., :3]
        return object_model

    def forward(self, imprint):
        img_embedding = self.autoencoder.encode(imprint)
        predicted_pose = self.pose_estimation_network(img_embedding)
        return predicted_pose

    def augmented_forward(self, imprint):
        # Augment the forward by using the reconstructed imprint instead.
        img_embedding = self.autoencoder.encode(imprint)
        imprint_reconstructed = self.autoencoder.decode(img_embedding)
        predicted_pose = self.forward(imprint_reconstructed)
        return predicted_pose

    def _step(self, batch, batch_idx, phase='train'):

        model_input = self.get_model_input(batch)
        ground_truth = self.get_model_output(batch)

        model_output = self.forward(*model_input)
        loss = self._compute_loss(model_output, *ground_truth)

        if self.autoencoder_augmentation:
            augmented_model_output = self.augmented_forward(*model_input)
            self._log_object_pose_images(obj_pose_pred=augmented_model_output[:self.num_to_log],
                                         obj_pose_gth=ground_truth[0][:self.num_to_log], phase='augmented_{}'.format(phase))
            augmented_loss = self._compute_loss(augmented_model_output, *ground_truth)
            self.log('{}_loss_original'.format(phase), loss)
            self.log('{}_loss_augmented'.format(phase), augmented_loss)
            loss = loss + augmented_loss

        # Log the results: -------------------------
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        self._log_object_pose_images(obj_pose_pred=model_output[:self.num_to_log], obj_pose_gth=ground_truth[0][:self.num_to_log], phase=phase)
        self._log_imprint(batch, batch_idx=batch_idx, phase=phase)
        return loss

    def _get_sizes(self):
        sizes = {}
        sizes.update(self.input_sizes)
        sizes['dyn_input_size'] = self._get_dyn_input_size(sizes)
        sizes['dyn_output_size'] = self._get_dyn_output_size(sizes)
        return sizes

    def get_input_keys(self):
        input_keys = ['imprint']
        return input_keys

    @abc.abstractmethod
    def get_model_output_keys(self):
        output_keys = ['object_pose']
        return output_keys

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def get_model_input(self, sample):
        input_key = self.get_input_keys()
        model_input = [sample[key] for key in input_key]
        model_input = tuple(model_input)
        return model_input

    def get_model_output(self, sample):
        output_keys = self.get_model_output_keys()
        model_output = [sample[key] for key in output_keys]
        model_output = tuple(model_output)
        return model_output

    def _compute_loss(self, obj_pose_pred, obj_pose_gth):
        # MSE Loss on position and orientation (encoded as aixis-angle 3 values)
        if self.loss_name == 'pose_loss':
            axis_angle_pred = obj_pose_pred[..., 3:]
            R_pred = batched_trs.axis_angle_to_matrix(axis_angle_pred)
            t_pred = obj_pose_pred[..., :3]
            axis_angle_gth = obj_pose_gth[..., 3:]
            R_gth = batched_trs.axis_angle_to_matrix(axis_angle_gth)
            t_gth = obj_pose_gth[..., :3]
            pose_loss = self.pose_loss(R_1=R_pred, t_1=t_pred, R_2=R_gth, t_2=t_gth)
            loss = pose_loss
        elif self.loss_name == 'mse':
            loss = self.mse_loss(obj_pose_pred, obj_pose_gth)
        else:
            raise NotImplementedError('Loss named {} not implemented yet.'.format(self.loss_name))
        return loss

    # AUX FUCTIONS -----------------------------------------------------------------------------------------------------

    def _load_autoencoder(self, load_version, data_path, load_epoch=None, load_step=None):
        Model = BubbleAutoEncoderModel
        model_name = Model.get_name()
        if load_epoch is None or load_step is None:
            version_chkp_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                             'version_{}'.format(load_version), 'checkpoints')
            checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                              os.path.isfile(os.path.join(version_chkp_path, f))]
            checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])
        else:
            checkpoint_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                           'version_{}'.format(load_version), 'checkpoints',
                                           'epoch={}-step={}.ckpt'.format(load_epoch, load_step))
        model = Model.load_from_checkpoint(checkpoint_path)

        return model

    def _log_imprint(self, batch, batch_idx, phase):
        if self.current_epoch == 0 and batch_idx == 0:
            imprint_t = batch['imprint'][:self.num_to_log]
            self.logger.experiment.add_image('imprint_{}'.format(phase), get_imprint_grid(imprint_t),
                                             self.global_step)
            if self.autoencoder_augmentation:
                reconstructed_imprint_t = self.autoencoder.decode(self.autoencoder.encode(imprint_t))
                self.logger.experiment.add_image('imprint_reconstructed_{}'.format(phase),
                                                 get_imprint_grid(reconstructed_imprint_t), self.global_step)

    def _log_object_pose_images(self, obj_pose_pred, obj_pose_gth, phase):
        grid = get_object_pose_images_grid(obj_pose_pred, obj_pose_gth, self.plane_normal)
        self.logger.experiment.add_image('pose_estimation_{}'.format(phase), grid, self.global_step)


class FakeICPApproximationModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imprint):
        fake_pose_shape = imprint.shape[:-3] + (6,)
        fake_pose = torch.zeros(fake_pose_shape, device=imprint.device, dtype=imprint.dtype) # encoded as axis-angle
        return fake_pose

    @classmethod
    def get_name(cls):
        return 'fake_icp_approximation_model'

    @property
    def name(self):
        return self.get_name()

