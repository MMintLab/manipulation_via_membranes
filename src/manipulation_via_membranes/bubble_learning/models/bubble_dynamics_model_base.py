import os
import torch
import abc


from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule
from manipulation_via_membranes.bubble_learning.models.bubble_autoencoder import BubbleAutoEncoderModel
from manipulation_via_membranes.bubble_learning.models.dynamics_model_base import DynamicsModelBase
from manipulation_via_membranes.bubble_learning.aux.visualization_utils.image_grid import get_batched_image_grid, get_imprint_grid


class BubbleDynamicsModelBase(DynamicsModelBase):
    def __init__(self, *args, load_autoencoder_version=0, num_imprints_to_log=40, **kwargs):
        # Note: the imprints to log are in rows of 8
        self.num_imprints_to_log = num_imprints_to_log
        super().__init__(*args, **kwargs)

        self.autoencoder = self._load_autoencoder(load_version=load_autoencoder_version, data_path=self.dataset_params['data_name'])
        self.autoencoder.freeze()
        self.img_embedding_size = self.autoencoder.img_embedding_size # load it from the autoencoder

        self.dyn_model = self._get_dyn_model()

        self.save_hyperparameters() # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'bubble_dynamics_model_base'

    @abc.abstractmethod
    def forward(self, imprint, wrench, object_model, pos, ori, action):
        pass

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        dyn_input_size = sizes['dyn_input_size']
        dyn_output_size = sizes['dyn_output_size']
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim]*self.num_fcs + [dyn_output_size]
        dyn_model = FCModule(sizes=dyn_model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return dyn_model

    def _step(self, batch, batch_idx, phase='train'):
        action = batch['action']

        model_input = self.get_model_input(batch)
        ground_truth = self.get_model_output(batch)

        model_output = self.forward(*model_input, action)

        loss = self._compute_loss(*model_output, *ground_truth)

        # Log the results: -------------------------
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # Log imprints
        self._log_imprints(batch=batch, model_output=model_output, batch_idx=batch_idx, phase=phase)
        return loss
    
    # Loading Functionalities: -----------------------------------------------------------------------------------------

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

    # AUX Functions: ---------------------------------------------------------------------------------------------------

    def _log_imprints(self, batch, model_output, batch_idx, phase):
        imprint_t = batch['init_imprint'][:self.num_imprints_to_log]
        imprint_next = batch['final_imprint'][:self.num_imprints_to_log]
        imprint_indx = self.get_model_output_keys().index('init_imprint')
        imprint_next_rec = model_output[imprint_indx][:self.num_imprints_to_log]
        predicted_grid = get_imprint_grid(imprint_next_rec * torch.max(imprint_next_rec) / torch.max(
            imprint_next))  # trasform so they are in the same range
        gth_grid = get_imprint_grid(imprint_next)
        if batch_idx == 0:
            if self.current_epoch == 0:
                self.logger.experiment.add_image('init_imprint_{}'.format(phase), get_imprint_grid(imprint_t),
                                                 self.global_step)
                self.logger.experiment.add_image('next_imprint_gt_{}'.format(phase), gth_grid, self.global_step)
            self.logger.experiment.add_image('next_imprint_predicted_{}'.format(phase), predicted_grid,
                                             self.global_step)
