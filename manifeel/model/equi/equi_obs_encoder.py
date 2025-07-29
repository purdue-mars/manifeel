import torch
import torch.nn.functional as F
from torchvision import models as vision_models
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
import diffusion_policy.model.vision.crop_randomizer as dmvc
from manifeel.model.base_nets import SpatialSoftmax
from manifeel.model.equi.equi_encoder import EquivariantResEncoder76Cyclic, EquivariantResEncoder230Cyclic
from manifeel.utils.transform_utils import quaternion_to_sixd

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class InHandEncoder76(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        net = vision_models.resnet18(norm_layer=Identity)
        self.resnet = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size//2)

    def forward(self, ih):
        batch_size = ih.shape[0]
        return self.spatial_softmax(self.resnet(ih)).reshape(batch_size, -1)

class InHandEncoder230(torch.nn.Module):
    def __init__(self, out_size):
        super().__init__()
        net = vision_models.resnet18(norm_layer=Identity)
        self.resnet = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.spatial_softmax = SpatialSoftmax([512, 8, 8], num_kp=out_size//2)

    def forward(self, ih):
        batch_size = ih.shape[0]
        return self.spatial_softmax(self.resnet(ih)).reshape(batch_size, -1)

class EquivariantObsEnc(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape_meta,
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        n_hidden=128,
        N=8,
        initialize=True,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.token_type = nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr])

        self.rgb_keys = [k for k, v in obs_shape_meta.items() if v["type"] == "rgb"]
        self.lowdim_keys = [k for k, v in obs_shape_meta.items() if v["type"] == "low_dim"]

        # Determine the active input modalities based on available keys
        has_obs = any(k in {"wrist", "front", "side"} for k in self.rgb_keys)
        has_ih = any(k in {"left_tactile_camera_taxim", "right_tactile_camera_taxim"} for k in self.rgb_keys)

        # Build the input type list dynamically
        input_type_list = []

        if has_obs:
            self.enc_obs = EquivariantResEncoder230Cyclic(obs_channel, self.n_hidden, initialize)
            input_type_list += self.n_hidden * [self.group.regular_repr]  # agentview

        if has_ih:
            self.enc_ih = InHandEncoder230(self.n_hidden).to(self.device)
            input_type_list += self.n_hidden * [self.group.trivial_repr]  # ih

        # Always include position, rotation (6D as 3x2), and z position
        input_type_list += 4 * [self.group.irrep(1)]  # pos_xy + rot6d
        input_type_list += 1 * [self.group.trivial_repr]  # z pos

        # Define the final encoder output layer
        self.enc_out = nn.Linear(
            nn.FieldType(self.group, input_type_list),
            self.token_type,
        )
        
        self.gTgc = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.crop_randomizer = dmvc.CropRandomizer(
            input_shape=obs_shape,
            crop_height=crop_shape[0],
            crop_width=crop_shape[1],
        )

        self.resize_shape = obs_shape

    def get6DRotation(self, quat):
        # input quat is in xyzw
        return quaternion_to_sixd(quat)
        
    def forward(self, nobs):
        obs = None
        ih = None

        for key in self.rgb_keys:
            if key in {"left_tactile_camera_taxim", "right_tactile_camera_taxim"} and key in nobs: 
                ih = nobs[key]
            elif key in {"wrist", "front", "side"} and key in nobs:
                obs = nobs[key]

        ee_state = nobs["state"]
        ee_pos = ee_state[..., :3]
        ee_quat = ee_state[..., 3:]  # x, y, z, w

        batch_size = ee_state.shape[0]
        t = ee_state.shape[1]

        # Rearrange and preprocess available modalities
        if obs is not None:
            obs = rearrange(obs, "b t c h w -> (b t) c h w")
            obs = F.interpolate(obs, 
                                size=(self.resize_shape[1], self.resize_shape[2]), 
                                mode='bilinear', 
                                align_corners=False)
            obs = self.crop_randomizer(obs)
            enc_out = self.enc_obs(obs).tensor.reshape(batch_size * t, -1)  # b d
        else:
            enc_out = None

        if ih is not None:
            ih = rearrange(ih, "b t c h w -> (b t) c h w")
            ih = F.interpolate(ih, 
                            size=(self.resize_shape[1], self.resize_shape[2]), 
                            mode='bilinear', 
                            align_corners=False)
            ih = self.crop_randomizer(ih)
            ih_out = self.enc_ih(ih)
        else:
            ih_out = None

        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_rot = self.get6DRotation(ee_quat)

        pos_xy = ee_pos[:, 0:2]
        pos_z = ee_pos[:, 2:3]

        feature_list = []

        if enc_out is not None:
            feature_list.append(enc_out)

        if ih_out is not None:
            feature_list.append(ih_out)

        feature_list.extend([
            pos_xy,
            ee_rot[:, 0:1],
            ee_rot[:, 3:4],
            ee_rot[:, 1:2],
            ee_rot[:, 4:5],
            ee_rot[:, 2:3],
            ee_rot[:, 5:6],
            pos_z
        ])

        features = torch.cat(feature_list, dim=1)

        features = nn.GeometricTensor(features, self.enc_out.in_type)
        out = self.enc_out(features).tensor
        return rearrange(out, "(b t) d -> b t d", b=batch_size)
