import torch
import torch.nn as nn
import pytorch_lightning as pl
import abc

from manipulation_via_membranes.bubble_learning.models.aux.fc_module import FCModule
from manipulation_via_membranes.bubble_learning.models.pointnet.pointnet_object_embedding import PointNetObjectEmbedding


class DynamicsModelBase(pl.LightningModule):
    def __init__(self, input_sizes, object_embedding_size=10, num_fcs=2, fc_h_dim=100,
                 skip_layers=None, lr=1e-4, dataset_params=None, load_norm=False, activation='relu',
                 freeze_object_module=True):
        super().__init__()
        self.input_sizes = input_sizes
        self.object_embedding_size = object_embedding_size
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation
        self.load_norm = load_norm
        self.freeze_object_module = freeze_object_module

        self.object_embedding_module = self._load_object_embedding_module(
            object_embedding_size=self.object_embedding_size, freeze=self.freeze_object_module)

        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters()  # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'dynamics_model_base'

    @property
    def name(self):
        return self.get_name()

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def _get_dyn_model(self):
        pass

    @abc.abstractmethod
    def _step(self, batch, batch_idx, phase='train'):
        pass

    def _get_sizes(self):
        sizes = {}
        sizes.update(self.input_sizes)
        sizes['dyn_input_size'] = self._get_dyn_input_size(sizes)
        sizes['dyn_output_size'] = self._get_dyn_output_size(sizes)
        return sizes

    @abc.abstractmethod
    def get_input_keys(self):
        pass

    @abc.abstractmethod
    def get_model_output_keys(self):
        pass

    @abc.abstractmethod
    def get_next_state_map(self):
        pass

    @abc.abstractmethod
    def _get_dyn_input_size(self, sizes):
        pass

    @abc.abstractmethod
    def _get_dyn_output_size(self, sizes):
        pass

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
        next_state_map = self.get_next_state_map()
        model_output = [sample[next_state_map[key]] for key in output_keys]
        model_output = tuple(model_output)
        return model_output

    def _get_dyn_model(self):
        sizes = self._get_sizes()
        dyn_input_size = sizes['dyn_input_size']
        dyn_output_size = sizes['dyn_output_size']
        dyn_model_sizes = [dyn_input_size] + [self.fc_h_dim] * self.num_fcs + [dyn_output_size]
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
        return loss

    # Loading Functionalities: -----------------------------------------------------------------------------------------

    def _load_object_embedding_module(self, object_embedding_size, freeze=True):
        pointnet_model = PointNetObjectEmbedding(obj_embedding_size=object_embedding_size, freeze_pointnet=freeze)
        # Expected input shape (BatchSize, NumPoints, NumChannels), where NumChannels=3 (xyz)
        return pointnet_model

