# Copyright 2025 EditMGT Team. All rights reserved.
import os
import math
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from typing import Any, Optional, Type, List
from collections import OrderedDict
from termcolor import cprint
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from peft.tuners.tuners_utils import BaseTunerLayer

class enable_lora:
    # for controling if lora is enable.
    def __init__(self, lora_modules: List[BaseTunerLayer], activated: bool) -> None:
        self.activated: bool = activated
        if activated:
            return
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]

    def __enter__(self) -> None:
        if self.activated:
            return

        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(0)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self.activated:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]

class LinearProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.initial_projection_dim = in_dim * 4
        self.projector = nn.Sequential(
            nn.Linear(in_dim, self.initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.initial_projection_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)

class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, time_embedding_dim: Optional[int] = None):
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, time_embedding_dim: Optional[int] = None
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        attn_output, attn_output_weights = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ):
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        latents = latents + self.mlp(self.ln_ff(latents, timestep_embedding))
        return latents


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim=None,
        input_dim=None,
        time_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(
            time_embedding_dim or width, width, bias=True
        )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None):
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding)
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents

class ConFluxAttnProcessor2_0:
    # Modified from diffusers.models.attention_processor import FluxAttnProcessor2_0
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.attention_weights = None
        self.capture_attention = False
        self.text_seq_len = 256
        self.block_id = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # --------- edit
        reference_image_hidden_states: Optional[torch.Tensor] = None,
        reference_image_rotary_emb: Optional[torch.Tensor] = None,
        lora_part_enable: bool = False,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        with enable_lora((attn.to_q, attn.to_k, attn.to_v), not lora_part_enable):
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # visualize attention weights
            text_len = encoder_hidden_states_query_proj.shape[2]
            image_len = query.shape[2]

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            # visualize attention weights
            # Single block: hidden_states is already concatenated [text+image]
            text_len = self.text_seq_len
            image_len = query.shape[2] - text_len

        if image_rotary_emb is not None:
            # cprint(f'{query.shape} {image_rotary_emb[0].shape, image_rotary_emb[1].shape}', 'red')
            # torch.Size([2, 8, 1280, 128]) (torch.Size([1280, 128]), torch.Size([1280, 128]))
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        if reference_image_hidden_states is not None:
            reference_query = attn.to_q(reference_image_hidden_states)
            reference_key = attn.to_k(reference_image_hidden_states)
            reference_value = attn.to_v(reference_image_hidden_states)
            reference_query = reference_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            reference_key = reference_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            reference_value = reference_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                reference_query = attn.norm_q(reference_query)
            if attn.norm_k is not None:
                reference_key = attn.norm_k(reference_key)

            if reference_image_rotary_emb is not None:
                reference_query = apply_rotary_emb(reference_query, reference_image_rotary_emb)
                reference_key = apply_rotary_emb(reference_key, reference_image_rotary_emb)
                
            # Only here is there an intersection. If set to None, it will have no effect on the result.
            query = torch.cat([query, reference_query], dim=2)
            key = torch.cat([key, reference_key], dim=2)
            value = torch.cat([value, reference_value], dim=2)
        
            if hasattr(attn, "reference_strength_factor"):  # to control the strength of raw image
                attention_mask = torch.zeros(
                    query.shape[2], key.shape[2], device=query.device, dtype=query.dtype
                )
                reference_n = reference_query.shape[2]
                bias = torch.log(attn.reference_strength_factor[0])
                attention_mask[-reference_n:, :-reference_n] = bias
                attention_mask[:-reference_n, -reference_n:] = bias
        else:
            reference_query, reference_key = None, None

        # ------------- visualize attention weights
        if self.capture_attention:
            if reference_query is not None and reference_key is not None:
                capture_query = query[:, :, :-reference_query.shape[2]]
                capture_key = key[:, :, :-reference_key.shape[2]]
                if hasattr(attn, "reference_strength_factor"):  # to control the strength of raw image
                    capture_seq_len = capture_query.shape[2]  # 1280
                    capture_attention_mask = attention_mask[:capture_seq_len, :capture_seq_len]
            else:
                capture_query = query
                capture_key = key

            if hasattr(attn, "reference_strength_factor") and capture_attention_mask is not None:
                # torch.Size([1280, 1280]) -> torch.Size([bs, head, 1280, 1280]) the same with 
                capture_attention_mask = capture_attention_mask.unsqueeze(0).unsqueeze(0)
                capture_attention_mask = capture_attention_mask.expand(capture_query.shape[0], capture_query.shape[1], -1, -1)
 
            scale = 1.0 / (head_dim ** 0.5)
            attention_scores = torch.matmul(capture_query, capture_key.transpose(-2, -1)) * scale
            if hasattr(attn, "reference_strength_factor") and capture_attention_mask is not None:
                attention_scores = attention_scores + capture_attention_mask
            
            attention_probs = F.softmax(attention_scores, dim=-1)
            text_to_image = attention_probs[:, :, :text_len, text_len:]
            image_to_text = attention_probs[:, :, text_len:, :text_len]
            image_to_image = attention_probs[:, :, text_len:, text_len:]
            text_to_text = attention_probs[:, :, :text_len, :text_len]

            # reference attention, if the text emb is not included, it will be meaningless
            if reference_query is not None and reference_key is not None:
                if encoder_hidden_states is not None:
                    ref_cap_query = torch.cat([query[:, :, :encoder_hidden_states.shape[2]], query[:, :, -reference_query.shape[2]:]], dim=2)
                    ref_cap_key = torch.cat([key[:, :, :encoder_hidden_states.shape[2]], key[:, :, -reference_query.shape[2]:]], dim=2)
                else:
                    ref_cap_query = query[:, :, :-reference_query.shape[2]]
                    ref_cap_key = key[:, :, :-reference_key.shape[2]]
                ref_cap_scores = torch.matmul(ref_cap_query, ref_cap_key.transpose(-2, -1)) * scale
                if hasattr(attn, "reference_strength_factor") and capture_attention_mask is not None:
                    ref_cap_scores = ref_cap_scores + capture_attention_mask
                ref_cap_probs = F.softmax(ref_cap_scores, dim=-1)
                ref_cap_text_to_image = ref_cap_probs[:, :, :text_len, text_len:]

            self.attention_weights = {
                'text_to_image': text_to_image.detach().cpu(),
                'image_to_text': image_to_text.detach().cpu(),
                'image_to_image':  image_to_image.detach().cpu(),
                'text_to_text':  text_to_text.detach().cpu(),
                'full_attention': attention_probs.detach().cpu(),
                'reference_text_to_image': ref_cap_text_to_image.detach().cpu() \
                                if reference_query is not None and reference_key is not None else None,
                'text_len': text_len,
                'image_len': image_len,
                'block_type': 'double' if encoder_hidden_states is not None else 'single',
                'block_id': self.block_id
            }
        # ------------- 

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            if reference_image_hidden_states is not None:
                encoder_hidden_states, hidden_states, reference_image_hidden_states = (
                    hidden_states[:, :encoder_hidden_states.shape[1]],
                    hidden_states[
                        :, encoder_hidden_states.shape[1]:-reference_image_hidden_states.shape[1]
                    ],
                    hidden_states[:, -reference_image_hidden_states.shape[1]:],
                )
            else:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1]:],
                )

            with enable_lora((attn.to_out[0],), not lora_part_enable):
                # linear proj
                hidden_states = attn.to_out[0](hidden_states)
                # dropout
                hidden_states = attn.to_out[1](hidden_states)

            if attn.to_add_out is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            if reference_image_hidden_states is not None:
                reference_image_hidden_states = attn.to_out[0](reference_image_hidden_states)
                reference_image_hidden_states = attn.to_out[1](reference_image_hidden_states)
                return hidden_states, encoder_hidden_states, reference_image_hidden_states
            else:
                return hidden_states, encoder_hidden_states
        else:
            if reference_image_hidden_states is not None:   
                hidden_states, reference_image_hidden_states = (
                    hidden_states[:, : -reference_image_hidden_states.shape[1]],
                    hidden_states[:, -reference_image_hidden_states.shape[1] :],
                )
                return hidden_states, reference_image_hidden_states
            else:
                return hidden_states

def save_checkpoint(args, accelerator, global_step, logger):
    output_dir = args.output_dir

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")


def prepare_cond_token(split_vae_encode, pixel_values, vq_model):
    batch_size = pixel_values.shape[0]
    
    split_batch_size = split_vae_encode if split_vae_encode is not None else batch_size
    num_splits = math.ceil(batch_size / split_batch_size)
    image_tokens = []
    for i in range(num_splits):
        start_idx = i * split_batch_size
        end_idx = min((i + 1) * split_batch_size, batch_size)
        image_tokens.append(
            vq_model.quantize(vq_model.encode(pixel_values[start_idx:end_idx]).latents)[2][2].reshape(
                split_batch_size, -1
            )
        )
    image_tokens = torch.cat(image_tokens, dim=0)

    return image_tokens