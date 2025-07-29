import copy
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

class DiffusionAnyTouchPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            anytouch_encoder: nn.Module,
            finetune = False,
            tactile_emb_dim= 512,
            use_same_patchemb=True,
            anytouch_encoder_embed_dim=None,
            anytouch_encoder_path = None,
            device = 'cuda:0',
            n_action_steps = 8,
            n_obs_steps=2,
            horizon = 16,
            num_inference_steps=None,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        ckpt = torch.load(anytouch_encoder_path, map_location='cpu')['model']
        anytouch_encoder = load_model_from_multi_clip(ckpt, anytouch_encoder, use_same_patchemb)
        self.anytouch_encoder = anytouch_encoder.to(device)
        print('anytouch encoder loaded')
        if not finetune:
            for param in self.anytouch_encoder.parameters():
                param.requires_grad = False
        if anytouch_encoder_embed_dim is None:
            raise ValueError("anytouch_encoder_embed_dim must be specified for anytouch representation")
        self.anytouch_head = nn.Linear(anytouch_encoder_embed_dim, tactile_emb_dim)
        
        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim
        # count how many tactile features are there
        global_cond_dim = obs_feature_dim + tactile_emb_dim
        global_cond_dim = global_cond_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.horizon = horizon

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if True else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.kwargs = kwargs
        self.n_obs_steps = n_obs_steps
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1)

        tactile_features = {}
        for key, value in nobs.items():
            if key in ['left_tactile_camera_taxim', 'tactile_0']:
                tactile = value
                B, _, _, _, _ = tactile.shape
                tactile = tactile[:, :To, ...]
                sensor_type = torch.zeros(B, device=device, dtype=torch.int64)
                tactile_feature = self.anytouch_encoder(tactile, sensor_type=sensor_type)
                # print(f"tactile feature output shape {tactile_feature.shape}")
                tactile_feature = self.anytouch_head(tactile_feature)
                tactile_feature = tactile_feature.reshape(B, -1)
                tactile_features[key] = tactile_feature

        for key, value in tactile_features.items():
            global_cond = torch.cat([global_cond, value], dim=1)
        
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        tactile_features = {}
        for key, value in nobs.items():
            if key in ['left_tactile_camera_taxim', 'tactile_0']:
                tactile = value
                B, _, _, _, _ = tactile.shape
                tactile = tactile[:, :self.n_obs_steps, ...]
                sensor_type = torch.zeros(B, device=self.device, dtype=torch.int64)
                tactile_feature = self.anytouch_encoder(tactile, sensor_type=sensor_type) # (b*t, 768)
                # print(f"tactile feature output shape {tactile_feature.shape}")
                tactile_feature = self.anytouch_head(tactile_feature)
                tactile_feature = tactile_feature.reshape(B, -1)
                tactile_features[key] = tactile_feature

        for key, value in tactile_features.items():
            global_cond = torch.cat([global_cond, value], dim=1)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

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
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
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
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    

def load_model_from_multi_clip(ckpt, model, use_same_patchemb):
    new_ckpt = {}
    for key, item in ckpt.items():
        # print(f"key {key}")
        if "touch_model" in key or "touch_projection" in key or "sensor_token" in key and "sensor_token_proj" not in key:
            new_ckpt[key.replace('touch_mae_model.','')] = copy.deepcopy(item)
        if use_same_patchemb:
            if "video_patch_embedding" in key:
                new_key = key.replace('touch_mae_model.','')
                new_ckpt[new_key.replace("video_patch_embedding","touch_model.embeddings.patch_embedding")] = copy.deepcopy(item)
    
    for k, v in model.named_parameters():
        # print(f"key in named parameters: {k}")
        if k not in new_ckpt.keys():
            new_ckpt[k] = v
    
    model.load_state_dict(new_ckpt, strict=True)

    return model