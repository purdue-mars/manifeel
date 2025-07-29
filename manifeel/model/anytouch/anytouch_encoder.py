import torch
from typing import Tuple, Union
from einops import rearrange
from torch import nn
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPVisionTransformer


class AnyTouchEncoder(nn.Module):
    def __init__(self, 
                 config_path, 
                 num_frames,
                 tube_size,
                 pooling='global',
                 use_sensor_token=False, 
                 use_same_patchemb=False):
        super(AnyTouchEncoder, self).__init__()

        # Load config
        config = AutoConfig.from_pretrained(config_path)
        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = tube_size
        self.hidden_size = config.vision_config.hidden_size
        self.use_sensor_token = use_sensor_token
        self.use_same_patchemb = use_same_patchemb

        if self.use_sensor_token:
            self.sensor_token = nn.Parameter(torch.zeros(10, 5, self.hidden_size))

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_projection = nn.Linear(self.hidden_size, 
                                          config.projection_dim, 
                                          bias=False)

        if self.use_same_patchemb:
            self.touch_model.embeddings.patch_embedding = nn.Conv3d(
                in_channels=config.vision_config.num_channels,
                out_channels=self.touch_model.embeddings.embed_dim,
                kernel_size=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
                stride=(3, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
                bias=False,
            )

        self.pooling = pooling

        self.touch_model.embeddings.forward = self.emb_forward
        self.touch_model.forward = self.touch_forward

    def emb_forward(self, pixel_values: torch.FloatTensor, noise=None, sensor_type=None) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(self.touch_model.embeddings.position_ids)

        embeddings = patch_embeds + pos_emb[:, 1:, :]

        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)

        if self.use_sensor_token:
            sensor_emb = self.sensor_token[sensor_type]
            embeddings = torch.cat([class_embeds, sensor_emb, embeddings], dim=1)
        else:
            embeddings = torch.cat([class_embeds, embeddings], dim=1)
        #embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def touch_forward(self, pixel_values, 
                      sensor_type=None, 
                      return_dict=False) -> Union[Tuple, BaseModelOutputWithPooling]:
        hidden_states = self.touch_model.embeddings(pixel_values, 
                                                sensor_type=sensor_type)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)

        encoder_outputs = self.touch_model.encoder(inputs_embeds=hidden_states, return_dict=True)
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.touch_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward(self, x, sensor_type=None):
        
        _, T, _, _, _ = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        sensor_type = sensor_type.repeat(T)
        
        if self.use_same_patchemb:
            x = x.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        
        # print(x.shape, sensor_type.shape)

        x = self.touch_model(x, 
                            sensor_type=sensor_type, 
                            return_dict=True)
        if self.pooling == 'cls':
            out = self.touch_projection(x.pooler_output)
        else:
            out = self.touch_projection(x.last_hidden_state)

        return out
