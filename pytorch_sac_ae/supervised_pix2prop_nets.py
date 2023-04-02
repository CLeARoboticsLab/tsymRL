import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder
from decoder import make_decoder

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class PropState(nn.Module):
    """MLP prop state network."""
    def __init__(
        self, obs_shape, prop_obs_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, prop_obs_shape)
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, detach_encoder=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        prop_state = self.trunk(obs)

        return prop_state

    # def log(self, L, step, log_freq=LOG_FREQ):
    #     if step % log_freq != 0:
    #         return

    #     for k, v in self.outputs.items():
    #         L.log_histogram('train_actor/%s_hist' % k, v, step)

    #     L.log_param('train_actor/fc1', self.trunk[0], step)
    #     L.log_param('train_actor/fc2', self.trunk[2], step)
    #     L.log_param('train_actor/fc3', self.trunk[4], step)


class pix2prop(object):
    """pixel to prop algorithm."""
    def __init__(
        self,
        obs_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        prop_state_lr=1e-3,
        prop_state_beta=0.9,
        prop_state_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        prop_obs_shape = 6
    ):
        self.device = device
        self.discount = discount
        self.encoder_tau = encoder_tau
        self.prop_state_update_freq = prop_state_update_freq

        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda

        self.prop_state = PropState(
            obs_shape, prop_obs_shape, hidden_dim, encoder_type,
            encoder_feature_dim,
            num_layers, num_filters
        ).to(device)

        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.prop_state.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.prop_state_optimizer = torch.optim.Adam(
            self.prop_state.parameters(), lr=prop_state_lr, betas=(prop_state_beta, 0.999)
        )


        self.train()

    def train(self, training=True):
        self.training = training
        self.prop_state.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    def update_prop_state(self, obs, true_prop_state, L, step):
        est_prop_state = self.prop_state(obs)

        # prop_state_loss = F.mse_loss(est_prop_state,
                                #  true_prop_state)
        # L.log('train_prop/loss', prop_state_loss, step)

        # Log where the error is remaining after we train
        total_diff = F.mse_loss(est_prop_state, true_prop_state, reduction='none')
        rel_err = torch.abs(total_diff/true_prop_state.detach())
        avg_rel_err = rel_err.mean(dim = 0)
        average_diff = total_diff.mean(dim = 0)
        L.log('train_prop/premean_abs_err_pos1', total_diff[:,0].sum(), step)
        L.log('train_prop/premean_abs_err_pos2', total_diff[:,1].sum(), step)
        L.log('train_prop/premean_abs_err_pos3', total_diff[:,2].sum(), step)
        L.log('train_prop/premean_abs_err_vel1', total_diff[:,3].sum(), step)
        L.log('train_prop/premean_abs_err_vel2', total_diff[:,4].sum(), step)

        prop_state_loss = F.mse_loss(est_prop_state, true_prop_state, reduction='none')[:,4].mean()
        L.log('train_prop/loss', prop_state_loss, step)


        # Optimize the prop state net
        self.prop_state_optimizer.zero_grad()
        prop_state_loss.backward()
        self.prop_state_optimizer.step()

        print("Epoch ", step + 1, " loss is ", prop_state_loss.detach().item())


        # self.critic.log(L, step)

    def update_decoder(self, obs, target_obs, L, step):
        h = self.prop_state.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, dual_buffer, L, step):


        obs, true_prop_state = dual_buffer.sample()

        self.update_prop_state(obs, true_prop_state, L, step)

        # utils.soft_update_params(
        #     self.prop_state.encoder, self.critic_target.encoder,
        #     self.encoder_tau
        # )

        self.update_decoder(obs, obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
