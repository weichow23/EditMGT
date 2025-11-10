# Copyright 2025 EditMGT Team. All rights reserved.
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import os
import json
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import LoraLoaderMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.resnet import Downsample2D, Upsample2D
from diffusers.models.normalization import RMSNorm
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
from peft import LoraConfig
from src.v2_utils import MLPProjector, LinearProjector, FusedMLPProjector, PerceiverResampler, ConFluxAttnProcessor2_0, enable_lora
from src.dataset_utils import get_encode_hidden_state_len

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def get_text_encoder_length(text_encoder_architecture, return_main=False):
    if text_encoder_architecture == "CLIP":
        return 77
    if text_encoder_architecture == "Gemma":
        return 256
    elif text_encoder_architecture in ["CLIP_Gemma2", 'CLIP_Gemma2-raw', 'CLIP_Qwen2.5', 
                                       "CLIP_Gemma3", 'CLIP_Gemma3-raw', "CLIP_Gemma1", 'CLIP_llama',
                                       "CLIP_T5-l", "CLIP_T5-xl", "CLIP_T5-xxl"]:
        if return_main:
            return 256
        else:
            return 77, 256
    else:
        raise ValueError(f"Unknown text encoder architecture: {text_encoder_architecture}")

def setup_lora_model(transformer, local_lora_dir):        
    config_path = os.path.join(local_lora_dir, "lora_config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    if 'target_modules' in config_dict and isinstance(config_dict['target_modules'], list):
        config_dict['target_modules'] = config_dict['target_modules']
    lora_config = LoraConfig(**config_dict)
    transformer.add_adapter(lora_config)

def load_lora_adapter(transformer, local_lora_dir, print_para_name=False):
    """Load pre-trained LoRA weights"""
    setup_lora_model(transformer, local_lora_dir)
    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(local_lora_dir)
    new_lora_state_dict = {k.replace("unet._orig_mod.", "").replace('.weight', '.default.weight'): v for k, v in lora_state_dict.items()}
    new_lora_state_dict = {k: v.to(dtype=transformer.dtype, device=transformer.device) 
                        for k, v in new_lora_state_dict.items()}
    return transformer

@maybe_allow_in_graph
class SingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = ConFluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        # ------- edit
        reference_image_hidden_states=None,
        reference_temb=None,
        reference_image_rotary_emb=None,
        lora_part_enable: Optional[bool] = False,
    ):
        residual = hidden_states  # torch.Size([2, 1280, 1024]) for either 1024 or 512
        with enable_lora((self.norm.linear, self.proj_mlp), not lora_part_enable):
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        if reference_image_hidden_states is not None:
            reference_residual = reference_image_hidden_states
            reference_norm_hidden_states, reference_gate = self.norm(reference_image_hidden_states, emb=reference_temb)
            reference_mlp_hidden_states = self.act_mlp(self.proj_mlp(reference_norm_hidden_states))  
            attn_output, reference_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                reference_image_hidden_states=reference_norm_hidden_states if reference_image_hidden_states is not None else None,
                reference_image_rotary_emb=reference_image_rotary_emb if reference_image_hidden_states is not None else None,
                lora_part_enable=lora_part_enable,
            )
        else:
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

        with enable_lora((self.proj_out,), not lora_part_enable):
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.proj_out(hidden_states)
            hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if reference_image_hidden_states is not None:
            reference_image_hidden_states = torch.cat([reference_attn_output, reference_mlp_hidden_states], dim=2)
            reference_gate = reference_gate.unsqueeze(1)
            reference_image_hidden_states = reference_gate * self.proj_out(reference_image_hidden_states)
            reference_image_hidden_states = reference_residual + reference_image_hidden_states
            return hidden_states, reference_image_hidden_states
        else:
            return hidden_states

@maybe_allow_in_graph
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6, is_last=False):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = ConFluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.is_last = is_last
        if not self.is_last:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.attn.to_add_out = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        reference_image_hidden_states=None,
        reference_temb=None,
        reference_image_rotary_emb=None, 
        lora_part_enable: Optional[bool] = False,
    ):
        with enable_lora((self.norm1.linear,), not lora_part_enable):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        if reference_image_hidden_states is not None:
            reference_norm_hidden_states, reference_gate_msa, reference_shift_mlp, reference_scale_mlp, reference_gate_mlp = self.norm1(
                reference_image_hidden_states, emb=reference_temb
            )
            attn_output, context_attn_output, reference_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                reference_image_hidden_states=reference_norm_hidden_states,
                reference_image_rotary_emb=reference_image_rotary_emb,
                lora_part_enable=lora_part_enable,
            )
        else:
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if reference_image_hidden_states is not None:
            reference_attn_output = reference_gate_msa.unsqueeze(1) * reference_attn_output
            reference_image_hidden_states = reference_image_hidden_states + reference_attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if reference_image_hidden_states is not None:
            reference_norm_hidden_states = self.norm2(reference_image_hidden_states)
            reference_norm_hidden_states = reference_norm_hidden_states * (1 + reference_scale_mlp[:, None]) + reference_shift_mlp[:, None]

        with enable_lora((self.ff.net[2], ), not lora_part_enable):
            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        if reference_image_hidden_states is not None:
            reference_ff_output = self.ff(reference_norm_hidden_states)
            reference_ff_output = reference_gate_mlp.unsqueeze(1) * reference_ff_output
            reference_image_hidden_states = reference_image_hidden_states + reference_ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if not self.is_last:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
                
        if reference_image_hidden_states is not None:
            return encoder_hidden_states, hidden_states, reference_image_hidden_states
        else:
            return encoder_hidden_states, hidden_states

class UVit2DConvEmbed(nn.Module):
    def __init__(self, in_channels, block_out_channels, vocab_size, elementwise_affine, eps, bias):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, in_channels)
        self.layer_norm = RMSNorm(in_channels, eps, elementwise_affine)
        self.conv = nn.Conv2d(in_channels, block_out_channels, kernel_size=1, bias=bias)

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = self.conv(embeddings)
        return embeddings

class ConvMlmLayer(nn.Module):
    def __init__(
        self,
        block_out_channels: int,
        in_channels: int,
        use_bias: bool,
        ln_elementwise_affine: bool,
        layer_norm_eps: float,
        codebook_size: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(block_out_channels, in_channels, kernel_size=1, bias=use_bias)
        self.layer_norm = RMSNorm(in_channels, layer_norm_eps, ln_elementwise_affine)
        self.conv2 = nn.Conv2d(in_channels, codebook_size, kernel_size=1, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        logits = self.conv2(hidden_states)
        return logits

class SwiGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function. It's similar to `GEGLU`
    but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)

class Simple_UVitBlock(nn.Module):
    def __init__(
        self,
        channels,
        ln_elementwise_affine,
        layer_norm_eps,
        use_bias,
        downsample: bool,
        upsample: bool,
    ):
        super().__init__()

        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
            )
        else:
            self.downsample = None

        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
            )
        else:
            self.upsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = False
    # Due to NotImplementedError: DDPOptimizer backend: Found a higher order op in the graph. This is not supported. Please turn off DDP optimizer using torch._dynamo.config.optimize_ddp=False. Note that this can cause performance degradation because there will be one bucket for the entire Dynamo graph. 
    # Please refer to this issue - https://github.com/pytorch/pytorch/issues/104674.
    _no_split_modules = ["TransformerBlock", "SingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False, # unused in our implementation
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        vocab_size: int = 8256,
        codebook_size: int = 8192,
        downsample: bool = False,
        upsample: bool = False,
        text_encoder_architecture: str = "CLIP",
        connector_type: str = 'none'   # None will not be save
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim # 64 * 16 = 1024 # 24 * 128 = 3072  # 6 * 128 = 768; 12 * 64 = 768

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.inner_dim  #self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.inner_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.gradient_checkpointing = False
        in_channels_embed = self.inner_dim  #768
        ln_elementwise_affine = True
        layer_norm_eps = 1e-06
        use_bias = False
        micro_cond_embed_dim = 1280

        self.embed = UVit2DConvEmbed(
            in_channels_embed, self.inner_dim, self.config.vocab_size, ln_elementwise_affine, layer_norm_eps, use_bias
        )
        self.mlm_layer = ConvMlmLayer(
            self.inner_dim, in_channels_embed, use_bias, ln_elementwise_affine, layer_norm_eps, self.config.codebook_size
        )
        self.cond_embed = TimestepEmbedding( #1024+1280=2304 -> 1024
            micro_cond_embed_dim + self.config.pooled_projection_dim, self.inner_dim, sample_proj_bias=use_bias
        )
        self.encoder_proj_layer_norm = RMSNorm(self.inner_dim, layer_norm_eps, ln_elementwise_affine)

        self.project_to_hidden_norm = RMSNorm(in_channels_embed, layer_norm_eps, ln_elementwise_affine)
        self.project_to_hidden = nn.Linear(in_channels_embed, self.inner_dim, bias=use_bias)
        self.project_from_hidden_norm = RMSNorm(self.inner_dim, layer_norm_eps, ln_elementwise_affine) 
        self.project_from_hidden = nn.Linear(self.inner_dim, in_channels_embed, bias=use_bias)
        
        self.down_block = Simple_UVitBlock(
            self.inner_dim,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            downsample,
            False,
        )
        self.up_block = Simple_UVitBlock(
            self.inner_dim,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            False,
            upsample=upsample,
        )

        self.connector_type = connector_type
        self.text_encoder_architecture = text_encoder_architecture
        if connector_type=='mlp':
            self.connector = MLPProjector(in_dim=joint_attention_dim, out_dim=self.inner_dim)
        elif connector_type=='qformer':
            self.connector = PerceiverResampler(
                width=self.inner_dim,
                layers=3,
                heads=8,
                num_latents=get_text_encoder_length(text_encoder_architecture, return_main=True),
                input_dim=joint_attention_dim,
                time_embedding_dim=1024,
            )
        elif connector_type=='linear':
            self.connector = LinearProjector(in_dim=joint_attention_dim, out_dim=self.inner_dim)
        elif connector_type=='fmlp':
            self.connector = FusedMLPProjector(in_dim=joint_attention_dim, out_dim=self.inner_dim)
        else:
            self.connector = None
            cprint('use pre-trained MGT model', 'cyan')
        
        self.text_seq_len = get_encode_hidden_state_len(self.text_encoder_architecture)
        self.capture_processors = {}

    def register_attention_hooks(self, target_blocks):
        """Replace the attention processor for all target blocks"""
        for block_idx in target_blocks:
            if block_idx < len(self.transformer_blocks):
                target_attn = self.transformer_blocks[block_idx].attn
                processor_name = f"transformer_blocks.{block_idx}.attn"
            else:
                single_block_idx = block_idx - len(self.transformer_blocks)
                if single_block_idx < len(self.single_transformer_blocks):
                    target_attn = self.single_transformer_blocks[single_block_idx].attn
                    processor_name = f"single_transformer_blocks.{single_block_idx}.attn"
                else:
                    print(f"Target block {block_idx} is out of range")
                    continue
            
            target_attn.processor.block_id = block_idx
            target_attn.processor.capture_attention = True
            target_attn.processor.text_seq_len = self.text_seq_len
            self.capture_processors[processor_name] = target_attn.processor
            
    def clear_attention_hooks(self):
        for processor in self.capture_processors.values():
            processor.capture_attention = False
            processor.attention_weights = None
            processor.block_id = None
        self.capture_processors.clear()

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples= None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        micro_conds: torch.Tensor = None,
        reference_image_hidden_states: torch.Tensor = None,
        reference_image_ids: torch.Tensor = None,
        reference_image_timestep: int=0,
        lora_part_enable: Optional[bool] = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.  [bs, 64, 64] for 1024 and [bs, 32, 32] for 512
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if self.connector_type is not None and self.connector_type != 'none':
            if self.connector_type == 'qformer':
                timestep_emb = get_timestep_embedding(timestep.to(hidden_states.dtype) * 1000, self.inner_dim)
                encoder_hidden_states = self.connector(
                    encoder_hidden_states, timestep_embedding=timestep_emb.unsqueeze(dim=1)
                )
            else:
                encoder_hidden_states = self.connector(encoder_hidden_states.to(self.dtype))
        
        micro_cond_encode_dim = 256 # same as self.config.micro_cond_encode_dim = 256 from amused
        micro_cond_embeds = get_timestep_embedding(
            micro_conds.flatten(), micro_cond_encode_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        ) # shape of micro_cond_embeds [10, 256]
        micro_cond_embeds = micro_cond_embeds.reshape((hidden_states.shape[0], -1)) # shape of micro_cond_embeds [2, 1280]  

        pooled_projections = torch.cat([pooled_projections, micro_cond_embeds], dim=1) # [2, 768] [2, 1280] -> [2, 2048] # [2, 1024] [2, 1280] -> [2, 2304]
        pooled_projections = pooled_projections.to(dtype=self.dtype)
        pooled_projections = self.cond_embed(pooled_projections).to(encoder_hidden_states.dtype)    

        encoder_hidden_states = self.context_embedder(encoder_hidden_states) # 1024 -> 1024 
        encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states) # to figure out how many parameters here

        hidden_states = self.embed(hidden_states) # output will be [2,768,16,16] # [2, 1024, 16, 16] 
        hidden_states = self.down_block(hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hidden_states = self.project_to_hidden_norm(hidden_states) # norm first or proj first?
        hidden_states = self.project_to_hidden(hidden_states)

        if reference_image_hidden_states is not None:
            reference_image_hidden_states = self.embed(reference_image_hidden_states)
            reference_image_hidden_states = self.down_block(reference_image_hidden_states)
            reference_batch_size, reference_channels, reference_height, reference_width = reference_image_hidden_states.shape
            reference_image_hidden_states = reference_image_hidden_states.permute(0, 2, 3, 1).reshape(reference_batch_size, reference_height * reference_width, reference_channels)
            reference_image_hidden_states = self.project_to_hidden_norm(reference_image_hidden_states) 
            reference_image_hidden_states = self.project_to_hidden(reference_image_hidden_states)

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        ) # self.inner_dim -> self.config.pooled_projection_dim

        if reference_image_hidden_states is not None:
            reference_temb = (
                self.time_text_embed(torch.ones_like(timestep) * reference_image_timestep * 1000, pooled_projections)
                if guidance is None
                else self.time_text_embed(
                    torch.ones_like(timestep) * reference_image_timestep * 1000, guidance, pooled_projections
                )
            )

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids) # a tuple
        
        if reference_image_ids is not None and reference_image_hidden_states is not None:
            reference_image_rotary_emb = self.pos_embed(reference_image_ids) 

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if reference_image_hidden_states is not None:
                    encoder_hidden_states, hidden_states, reference_image_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states=hidden_states,
                        reference_image_hidden_states=reference_image_hidden_states if reference_image_hidden_states is not None else None,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,  
                        reference_temb = reference_temb if reference_image_hidden_states is not None else None,
                        image_rotary_emb=image_rotary_emb,
                        reference_image_rotary_emb=reference_image_rotary_emb if reference_image_hidden_states is not None else None,
                        lora_part_enable=lora_part_enable,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

            else:
                if reference_image_hidden_states is not None:
                    encoder_hidden_states, hidden_states, reference_image_hidden_states = block(
                        hidden_states=hidden_states,
                        reference_image_hidden_states=reference_image_hidden_states if reference_image_hidden_states is not None else None,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        reference_temb = reference_temb if reference_image_hidden_states is not None else None,
                        image_rotary_emb=image_rotary_emb,
                        reference_image_rotary_emb=reference_image_rotary_emb if reference_image_hidden_states is not None else None,
                        lora_part_enable=lora_part_enable,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,  
                        image_rotary_emb=image_rotary_emb,
                    )
                
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                if reference_image_hidden_states is not None:
                    hidden_states, reference_image_hidden_states = torch.utils.checkpoint.checkpoint(
                        hidden_states=hidden_states,
                        reference_image_hidden_states=reference_image_hidden_states if reference_image_hidden_states is not None else None,
                        temb=temb,
                        reference_temb = reference_temb if reference_image_hidden_states is not None else None,
                        image_rotary_emb=image_rotary_emb,
                        reference_image_rotary_emb=reference_image_rotary_emb if reference_image_hidden_states is not None else None,
                        lora_part_enable=lora_part_enable,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
            else:
                if reference_image_hidden_states is not None:
                    hidden_states, reference_image_hidden_states = block(
                        hidden_states=hidden_states,
                        reference_image_hidden_states=reference_image_hidden_states if reference_image_hidden_states is not None else None,
                        temb=temb,
                        reference_temb = reference_temb if reference_image_hidden_states is not None else None,
                        image_rotary_emb=image_rotary_emb,
                        reference_image_rotary_emb=reference_image_rotary_emb if reference_image_hidden_states is not None else None,
                        lora_part_enable=lora_part_enable,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                    )

            # controlnet residual
            if controlnet_single_block_samples is not None: 
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control][:, encoder_hidden_states.shape[1] :, ...]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...] 

        hidden_states = self.project_from_hidden_norm(hidden_states)
        hidden_states = self.project_from_hidden(hidden_states)

        #  torch.Size([2, 1024, 32, 32]) for either 1024 or 512
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        # torch.Size([2, 1024, 64, 64]) for 1024; torch.Size([2, 1024, 32, 32]) for 512
        hidden_states = self.up_block(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        output = self.mlm_layer(hidden_states)

        if not return_dict:
            return (output,)

        return output
