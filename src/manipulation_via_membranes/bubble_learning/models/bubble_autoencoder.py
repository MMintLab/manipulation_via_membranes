import torch
import torch.nn as nn

from manipulation_via_membranes.bubble_learning.models.old.bubble_dynamics_residual_model import BubbleDynamicsResidualModel


class BubbleAutoEncoderModel(BubbleDynamicsResidualModel):

    def __init__(self, *args, reconstruct_key='delta_imprint', **kwargs):
        self.reconstruct_key = reconstruct_key
        super().__init__(*args, **kwargs)
        self.batch_norm = nn.BatchNorm2d(2)
        self.save_hyperparameters() # Important! Every model extension must add this line!

    @classmethod
    def get_name(cls):
        return 'bubble_autoencoder_model'

    def _get_dyn_model(self):
        return None # We do not have a dyn model in this case

    def forward(self, imprint):
        imprint_emb = self.img_encoder(imprint)
        imprint_reconstructed = self.img_decoder(imprint_emb)
        return imprint_reconstructed

    def encode(self, imprint):
        # method to query the model once trained.
        # It normalizes the imprint before embedding.
        imprint_norm = self._norm_imprint(imprint)
        imprint_emb = self.img_encoder(imprint_norm)
        return imprint_emb

    def decode(self, imprint_emb):
        # method to query the model once trained.
        # It decodes and denormalizes the imprint to obtain a good reconstruction.
        imprint_rec_norm = self.img_decoder(imprint_emb)
        imprint_rec = self._denorm_imprint(imprint_rec_norm)
        return imprint_rec

    def _norm_imprint(self, imprint):
        mean = self.batch_norm._buffers['running_mean']
        var = self.batch_norm._buffers['running_var']
        eps = self.batch_norm.eps
        gamma = self.batch_norm._parameters['weight']
        beta = self.batch_norm._parameters['bias']

        norm_imprint_r = (imprint.swapaxes(1,3)-mean)/torch.sqrt(var+eps)*gamma + beta
        norm_imprint = norm_imprint_r.swapaxes(1,3) # swap axes back
        return norm_imprint

    def _denorm_imprint(self, norm_imprint):
        mean = self.batch_norm._buffers['running_mean']
        var = self.batch_norm._buffers['running_var']
        eps = self.batch_norm.eps
        gamma = self.batch_norm._parameters['weight']
        beta = self.batch_norm._parameters['bias']
        imprint_r = (norm_imprint.swapaxes(1,3)-beta)/gamma*torch.sqrt(var+eps)+mean
        imprint = imprint_r.swapaxes(1,3) # swap axes back
        return imprint

    def _get_sizes(self):
        imprint_size = self.input_sizes[self.reconstruct_key]
        sizes = {'imprint': imprint_size}
        return sizes

    def _step(self, batch, batch_idx, phase='train'):
        imprint = batch[self.reconstruct_key]
        imprint_t = self.batch_norm(imprint) # Normalize the input for both the model and loss
        imprint_rec = self.forward(imprint_t)
        loss = self._compute_loss(imprint_t, imprint_rec)
        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)
        # add image:
        if batch_idx == 0:
            reconstructed_grid = self._get_image_grid(imprint_rec*torch.max(imprint_rec)/torch.max(imprint_t))
            gth_grid = self._get_image_grid(imprint_t)
            self.logger.experiment.add_image('{}_reconstructed_{}'.format(self.reconstruct_key, phase), reconstructed_grid, self.global_step)
            denorm_rec_imprint = self._denorm_imprint(imprint_rec)
            denorm_rec_grid = self._get_image_grid(denorm_rec_imprint*torch.max(denorm_rec_imprint)/torch.max(imprint))
            self.logger.experiment.add_image('{}_denorm_reconstructed_{}'.format(self.reconstruct_key, phase), denorm_rec_grid, self.global_step)
            if self.current_epoch == 0:
                self.logger.experiment.add_image('{}_gth_{}'.format(self.reconstruct_key, phase), gth_grid, self.global_step)
                denorm_imprint = self._denorm_imprint(imprint_t)
                self.logger.experiment.add_image('{}_unnorm_gth_{}'.format(self.reconstruct_key, phase), self._get_image_grid(denorm_imprint), self.global_step)
                self.logger.experiment.add_image('{}_original_gth_{}'.format(self.reconstruct_key, phase), self._get_image_grid(imprint), self.global_step)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _compute_loss(self, imprint_rec, imprint_d_gth):
        imprint_reconstruction_loss = self.mse_loss(imprint_rec, imprint_d_gth)
        loss = imprint_reconstruction_loss

        return loss
