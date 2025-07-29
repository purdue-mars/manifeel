from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops
from equi_diffpo.model.common.module_attr_mixin import ModuleAttrMixin
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.model.diffusion.dp3_conditional_unet1d import ConditionalUnet1D
from equi_diffpo.model.diffusion.mask_generator import LowdimMaskGenerator
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.model.vision.pointnet_extractor import DP3Encoder
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *

class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
    
class FMDP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                 img_crop_shape=crop_shape,
                                 out_channel=encoder_output_dim,
                                 pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                 use_pc_color=use_pc_color,
                                 pointnet_type=pointnet_type,)

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )


        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        sigma = 0.0
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        # print_params(self)

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            condition_data_pc=None,
            condition_mask_pc=None,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        pred_horizon = 16
        device = self.device
        action_dim = self.action_dim

        timehorion = self.num_inference_steps

        for t in range(timehorion):
            noise = torch.rand(1, pred_horizon, action_dim).to(device)
            x0 = noise.expand(condition_data.shape[0], -1, -1)
            timestep = torch.tensor([t / timehorion]).to(device)

            if t == 0:
                vt = self.model(x0,
                                timestep,
                                global_cond=global_cond)
                traj = (vt * 1 / timehorion + x0)

            else:
                vt = self.model(traj,
                                timestep,
                                global_cond=global_cond)
                traj = (vt * 1 / timehorion + traj)

        return traj


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        if 'robot0_eye_in_hand_image' in obs_dict:
            del obs_dict['robot0_eye_in_hand_image']
        if 'agentview_image' in obs_dict:
            del obs_dict['agentview_image']
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        
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

        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        ret = self.obs_encoder(this_nobs)

        final_feat = ret['final_feat']

        # reshape back to B, Do
        global_cond = final_feat.reshape(B, -1)
        # global_cond = nobs_features
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
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        
        # get prediction
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        if 'robot0_eye_in_hand_image' in batch['obs']:
            del batch['obs']['robot0_eye_in_hand_image']
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        B, T = nactions.shape[:2]
        To = 2
        # handle different ways of passing observation
        global_cond = None
        trajectory = nactions
        
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        ret = self.obs_encoder(this_nobs, mode="train")

        final_feat = ret['final_feat']

        # reshape back to B, Do
        global_cond = final_feat.reshape(B, -1)  # [B, To*Do]

        # Sample noise that we'll add to the images
        x0 = torch.randn(trajectory.shape, device=trajectory.device)
        timestep, xt, ut = self.FM.sample_location_and_conditional_flow(x0, trajectory)

        # Predict the noise residual        
        vt = self.model(sample=xt, 
                          timestep=timestep, 
                          global_cond=global_cond)

        loss = F.mse_loss(vt, ut, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict