# Copyright 2025 EditMGT Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from termcolor import cprint
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models import VQModel
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from transformers import AutoModel, AutoTokenizer
from src.scheduler import Scheduler
from src.local_utils import smooth_local_scores, tokens_to_attn, rescale_scores
from src.transformer import Transformer2DModel, get_text_encoder_length
from src.dataset_utils import tokenize_prompt, encode_prompt, process_image
from src.v2_utils import prepare_cond_token

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

def _prepare_latent_image_ids(height, width, device, dtype, up_sample=False):
    if up_sample:
        height = 2 * height
        width = 2 * width
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)

def pilimage_to_token(pipe, image):
    latents = pipe.vqvae.encode(image.to(dtype=pipe.vqvae.dtype, device=pipe._execution_device)).latents
    latents_bsz, channels, latents_height, latents_width = latents.shape
    latents = pipe.vqvae.quantize(latents)[2][2].reshape(latents_bsz, latents_height, latents_width)
    return latents

class Pipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer or List
    text_encoder: CLIPTextModelWithProjection or List
    transformer: Transformer2DModel
    scheduler: Scheduler
    tokenizer_t5: AutoTokenizer
    text_encoder_t5: AutoModel

    model_cpu_offload_seq = "text_encoder->text_encoder_t5->transformer->vqvae"

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: Transformer2DModel,
        scheduler: Scheduler,
        tokenizer_t5: AutoTokenizer,
        text_encoder_t5: AutoModel,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            tokenizer_t5=tokenizer_t5,
            text_encoder_t5=text_encoder_t5,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor,
                                                  do_normalize=False)

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
            do_resize=True,
        )

        # [1,2] is to capture the weight of the transformer block with idx 1 and 2
        self.attention_enable_blocks = None 
        # Local attention threshold [0, 1]
        self.local_guidance=0
        
        self.current_attention_list = {}
        self.selected_mask_token = {}
        self.local_scores_list = {}
        self.local_query_text = None
 
    def safe_decode_latents(self, latents, batch_size, height, width, output_type):
        decode_latents = latents.clone()
        mask_token_id = self.scheduler.config.mask_token_id
        # Replace the mask token with 0 (pure black)
        decode_latents[decode_latents == mask_token_id] = 0
        
        needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast
        if needs_upcasting:
            self.vqvae.float()
        
        decoded = self.vqvae.decode(
            decode_latents,
            force_not_quantize=True,
            shape=(
                batch_size,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
                self.vqvae.config.latent_channels,
            ),
        ).sample.clip(0, 1)
        decoded = self.image_processor.postprocess(decoded, output_type)
        
        if needs_upcasting:
            self.vqvae.half()
        
        return decoded
        
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 12,
        guidance_scale: float = 10.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        output_type="pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        micro_conditioning_aesthetic_score: int = 6,
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        mask_image: PipelineImageInput = None,  # inpaint
        return_target_step_images: List[int] = None,   # See attention and intermdiate image: Which steps to decode
        reference_image: PipelineImageInput = None,
        reference_strength: float = 1,
        lora_part_enable: Optional[bool] = False,       
        lora_scale: Optional[float] = None,           
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 16):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.IntTensor`, *optional*):
                Pre-generated tokens representing latent vectors in `self.vqvae`, to be used as inputs for image
                gneration. If not provided, the starting latents will be completely masked.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                Analogous to `encoder_hidden_states` for the positive prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                The targeted aesthetic score according to the laion aesthetic classifier. See
                https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                The targeted height, width crop coordinates. See the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                Configures the temperature scheduler on `self.scheduler` see `Scheduler#set_timesteps`.
        Examples:

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """

        if self.attention_enable_blocks is not None:
            self.transformer.register_attention_hooks(self.attention_enable_blocks)
        self.current_attention_list = {}
        self.selected_mask_token = {}
        self.local_scores_list = {}

        if (prompt_embeds is not None and encoder_hidden_states is None) or (
            prompt_embeds is None and encoder_hidden_states is not None
        ):
            raise ValueError("pass either both `prompt_embeds` and `encoder_hidden_states` or neither")

        if (negative_prompt_embeds is not None and negative_encoder_hidden_states is None) or (
            negative_prompt_embeds is None and negative_encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
            )

        if (prompt is None and prompt_embeds is None) or (prompt is not None and prompt_embeds is not None):
            raise ValueError("pass only one of `prompt` or `prompt_embeds`")

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        if height is None:
            height = self.transformer.config.sample_size * self.vae_scale_factor

        if width is None:
            width = self.transformer.config.sample_size * self.vae_scale_factor

        if prompt_embeds is None:
            encoder_hidden_states, prompt_embeds = encode_prompt(
                    [self.text_encoder, self.text_encoder_t5] if self.text_encoder_t5 is not None else self.text_encoder,
                    tokenize_prompt([self.tokenizer, self.tokenizer_t5] if self.tokenizer_t5 is not None else self.tokenizer, prompt, 
                                    self.transformer.text_encoder_architecture, device=self._execution_device), 
                    self.transformer.text_encoder_architecture
                )

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]

                negative_encoder_hidden_states, negative_prompt_embeds = encode_prompt(
                    [self.text_encoder, self.text_encoder_t5] if self.text_encoder_t5 is not None else self.text_encoder,
                    tokenize_prompt([self.tokenizer, self.tokenizer_t5] if self.tokenizer_t5 is not None else self.tokenizer, negative_prompt, 
                                    self.transformer.text_encoder_architecture, device=self._execution_device), 
                    self.transformer.text_encoder_architecture
                )        

            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1)
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)
            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])

        # Note that the micro conditionings _do_ flip the order of width, height for the original size
        # and the crop coordinates. This is how it was done in the original code base
        micro_conds = torch.tensor(
            [
                width,
                height,
                micro_conditioning_crop_coord[0],
                micro_conditioning_crop_coord[1],
                micro_conditioning_aesthetic_score,
            ],
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 else batch_size, -1)

        self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)

        starting_mask_ratio = 1.0

        if mask_image is not None:
            mask = self.mask_processor.preprocess(
                mask_image, height // self.vae_scale_factor, width // self.vae_scale_factor
            )

        shape = (batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)
        latents = torch.full(
            shape, self.scheduler.config.mask_token_id, dtype=torch.long, device=self._execution_device
        )  # 1024: torch.Size([bs, 64, 64])  512: torch.Size([bs, 32, 32])
        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order

        if reference_image is not None: 
            if isinstance(reference_image, list):
                reference_image = [process_image(img, size=height)['image'].to(dtype=self.vqvae.dtype, device=self._execution_device).unsqueeze(0) for img in reference_image]
                reference_image = torch.cat(reference_image, dim=0)
            else:
                reference_image = process_image(reference_image, size=height)['image'].to(dtype=self.vqvae.dtype, device=self._execution_device) 
                reference_image = reference_image.unsqueeze(0)

            reference_image_hidden_states = prepare_cond_token(split_vae_encode=None, pixel_values=reference_image, vq_model=self.vqvae)
            reference_image_hidden_states = reference_image_hidden_states.reshape(-1, height // self.vae_scale_factor, width // self.vae_scale_factor) # 处理和训练保持了一致

            if mask_image is not None:
                latents = reference_image_hidden_states
                mask = mask.reshape(mask.shape[0], latents.shape[-2], latents.shape[-1]).bool().to(latents.device)
                latents[mask] = self.scheduler.config.mask_token_id  # replace the mask region needs update
                starting_mask_ratio = mask.sum() / latents.numel()
            
        if reference_strength != 1:
            for name, module in self.transformer.named_modules():
                if not name.endswith(".attn"):
                    continue
                module.reference_strength_factor = torch.ones(1, 1) * reference_strength

        intermediate_latents = {}
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            timesteps_iter = enumerate(self.scheduler.timesteps)

            for i, timestep in timesteps_iter:
                if guidance_scale > 1.0:
                    model_input = torch.cat([latents] * 2)
                    if reference_image is not None:
                        reference_input = torch.cat([reference_image_hidden_states, reference_image_hidden_states], dim=0)
                else:
                    model_input = latents
                    if reference_image is not None:
                        reference_input = reference_image_hidden_states
                    
                img_ids = _prepare_latent_image_ids(model_input.shape[-2], model_input.shape[-1],
                                                    model_input.device, model_input.dtype, up_sample=height!=1024) # torch.Size([1024, 3])
                
                if '_' not in self.transformer.text_encoder_architecture:
                    txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = encoder_hidden_states.device, dtype = encoder_hidden_states.dtype)
                else:
                    txt_ids = torch.zeros(get_text_encoder_length(self.transformer.text_encoder_architecture, return_main=True),3).to(device = encoder_hidden_states.device, dtype = encoder_hidden_states.dtype)
                
                # image edit
                if reference_image is not None:
                    reference_image_ids = _prepare_latent_image_ids(reference_input.shape[-2], reference_input.shape[-1], 
                                                            reference_input.device, reference_input.dtype, up_sample=height!=1024)
                    model_output = self.transformer(
                        hidden_states = model_input,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        img_ids = img_ids,
                        txt_ids = txt_ids,
                        timestep = torch.tensor([timestep]*model_input.shape[0], device=model_input.device, dtype=torch.long),
                        reference_image_hidden_states=reference_input.to(dtype=model_input.dtype, device=model_input.device),
                        reference_image_ids=reference_image_ids,
                        joint_attention_kwargs={"scale": lora_scale},
                        lora_part_enable=lora_part_enable,
                    )
                else:
                    # text to image
                    model_output = self.transformer(
                        hidden_states = model_input,
                        micro_conds=micro_conds,
                        pooled_projections=prompt_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        img_ids = img_ids,
                        txt_ids = txt_ids,
                        timestep = torch.tensor([timestep]*model_input.shape[0], device=model_input.device, dtype=torch.long),
                    )

                if guidance_scale > 1.0:
                    uncond_logits, cond_logits = model_output.chunk(2)  # uncond comes before concat
                    model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

                current_attention, local_stacked, _ = tokens_to_attn(pipe=self, global_enable=False, prompt=prompt)

                local_scores = rescale_scores(local_stacked) # [bs, seq]
                local_scores = smooth_local_scores(local_scores, method='adaptive', strength=1.0) # smoothen

                latents = self.scheduler.step(
                    model_output=model_output,  # torch.Size([1, 8192, 32, 32])
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                    starting_mask_ratio=starting_mask_ratio,
                    # local
                    local_guidance=self.local_guidance,
                    local_scores=local_scores, # [batch, seq]
                    ref_latents=reference_image_hidden_states if reference_image is not None else None, # torch.Size([1, 32, 32])
                ).prev_sample   # torch.Size([1, 32, 32])

                # See Attention: Decode intermediate images at target steps
                if return_target_step_images is not None:
                    step_idx = i
                    if step_idx in return_target_step_images:
                        intermediate_latents[step_idx] = latents
                        self.current_attention_list[step_idx] = current_attention
                        self.selected_mask_token[step_idx] = self.scheduler.mask_selected
                        self.local_scores_list[step_idx] = local_scores

                if (i == len(self.scheduler.timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                )):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, timestep, latents)
        
        if output_type == "latent":
            output = latents
        else:
            output = self.safe_decode_latents(latents=latents, batch_size=batch_size, height=height, width=width, output_type=output_type)
        
        # See Attention: Decode intermediate images at target steps
        if return_target_step_images:
            intermediate_output = [self.safe_decode_latents(latents=l, batch_size=batch_size, height=height, 
                                                            width=width, output_type=output_type) for l in intermediate_latents.values()]

        self.maybe_free_model_hooks()

        # See Attention: Return results with intermediate images if requested
        if return_target_step_images:
            if not return_dict:
                return (output, intermediate_output)
            return ImagePipelineOutput(output), ImagePipelineOutput(intermediate_output)
        else:
            if not return_dict:
                return (output,)
            return ImagePipelineOutput(output)
        