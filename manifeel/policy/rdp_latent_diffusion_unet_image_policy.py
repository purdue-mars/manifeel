from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from manifeel.policy.rdp_diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from manifeel.model.vision.timm_obs_encoder import TimmObsEncoder
from manifeel.model.vae.model import VAE
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply

class LatentDiffusionUnetImagePolicy(DiffusionUnetImagePolicy):
    def __init__(self,
                 at: VAE,
                 use_latent_action_before_vq: bool,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 obs_encoder: Union[MultiImageObsEncoder, TimmObsEncoder],
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,  # dummy
                 n_groups=8,
                 cond_predict_scale=True,
                 # parameters passed to step
                 **kwargs):

        if at.use_vq:
            shape_meta['action']['shape'] = [at.n_latent_dims]
        else:
            shape_meta['action']['shape'] = [at.n_embed]

        original_horizon = horizon
        if at.use_conv_encoder:
            horizon = at.downsampled_input_h
            kernel_size = 3
        else:
            # hack: latent action can be viewed as a sequence of latent actions with horizon 1
            horizon = at.downsampled_input_h
            kernel_size = 1

        super().__init__(shape_meta,
                         noise_scheduler,
                         obs_encoder,
                         horizon,
                         n_action_steps,
                         n_obs_steps,
                         num_inference_steps,
                         obs_as_global_cond,
                         diffusion_step_embed_dim,
                         down_dims,
                         kernel_size,
                         n_groups,
                         cond_predict_scale,
                         **kwargs)

        self.original_horizon = original_horizon

        if horizon == 1:
            # hack: remove the upsample/downsample layers
            for module_list in self.model.down_modules:
                module_list[-1] = nn.Identity()

            for module_list in self.model.up_modules:
                module_list[-1] = nn.Identity()

        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.at.set_normalizer(normalizer)

    def to(self, device):
        super().to(device)
        self.at.to(device)
        return self

    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       dataset_obs_temporal_downsample_ratio: int,
                       extended_obs_dict: Dict[str, torch.Tensor] = None,
                       return_latent_action=False
                       ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            raise NotImplementedError

        # run sampling
        nlatent_sample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        # unnormalize prediction
        nlatent_sample = self.normalizer['latent_action'].unnormalize(nlatent_sample)

        # decode latent action
        # note: handle latent action correctly
        nlatent_sample = einops.rearrange(nlatent_sample, 'N T A -> N (T A)')
        if self.at.use_vq:
            if self.use_latent_action_before_vq:
                state_vq, _, _ = self.at.quant_state_with_vq(nlatent_sample)
            else:
                state_vq = nlatent_sample
        else:
            state_vq = nlatent_sample
            state_vq = self.at.postprocess_quant_state_without_vq(state_vq)

        if return_latent_action:
            action_pred = state_vq.unsqueeze(1).expand(-1, self.original_horizon, -1)
        else:
            if self.at.use_rnn_decoder:
                temporal_cond = self.at.get_temporal_cond(extended_obs_dict)
                temporal_cond = temporal_cond.to(self.device)
                nsample = self.at.get_action_from_latent_with_temporal_cond(state_vq, temporal_cond)
            else:
                nsample = self.at.get_action_from_latent(state_vq)

            # unnormalize action
            naction_pred = nsample[..., :Da]
            action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # hack: align with the training process
        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def predict_from_latent_action(self, latent_action: torch.Tensor, extended_obs_dict: Dict[str, torch.Tensor], extended_obs_last_step: int, dataset_obs_temporal_downsample_ratio: int, extend_obs_pad_after: bool = False):
        Da = self.action_dim
        To = self.n_obs_steps

        latent_action = latent_action.to(self.device)
        if self.at.use_rnn_decoder:
            if extend_obs_pad_after:
                extend_obs_pad_after_n = self.original_horizon - extended_obs_last_step
            else:
                extend_obs_pad_after_n = None
            print(f"extended_obs_last_step: {extended_obs_last_step}")
            print(f"extend_obs_pad_after_n: {extend_obs_pad_after_n}")
            temporal_cond = self.at.get_temporal_cond(extended_obs_dict, extended_obs_last_step, extend_obs_pad_after_n=extend_obs_pad_after_n)
            temporal_cond = temporal_cond.to(self.device)
            nsample = self.at.get_action_from_latent_with_temporal_cond(latent_action, temporal_cond)
        else:
            nsample = self.at.get_action_from_latent(latent_action)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # hack: align with the training process
        To = self.n_obs_steps * dataset_obs_temporal_downsample_ratio
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }

        return result


    # ========= training  ============
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # get latent action
        batch_size = nactions.shape[0]
        nlatent_actions = self.at.encoder(
            self.at.preprocess(nactions / self.at.act_scale)
        )
        # note: handle latent action correctly
        if self.at.use_vq:
            if not self.use_latent_action_before_vq:
                nlatent_actions, _, _ = self.at.quant_state_with_vq(nlatent_actions)
            else:
                if self.at.use_conv_encoder:
                    nlatent_actions = einops.rearrange(nlatent_actions, 'N T A -> N (T A)')
        else:
            nlatent_actions, _ = self.at.quant_state_without_vq(nlatent_actions)
        nlatent_actions = einops.rearrange(nlatent_actions, 'N (T A) -> N T A', T=self.horizon)
        nlatent_actions = self.normalizer['latent_action'].normalize(nlatent_actions)

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nlatent_actions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                   lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            raise NotImplementedError

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
