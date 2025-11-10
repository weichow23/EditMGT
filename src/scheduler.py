# Copyright 2025 The EditMGT Team. All rights reserved.
import math
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from termcolor import cprint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin

def gumbel_noise(t, generator=None):
    device = generator.device if generator is not None else t.device
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    '''
        mask_len: torch.size([bs,1])
    '''
    # Find the mask_len tokens with the lowest confidence and mask them. 
    # Because of the need for randomness, the temperature is 1.
    confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values  # Sort from low to high
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


@dataclass
class SchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor = None

def upsample_score_map(local_map):
    from einops import rearrange, repeat
    import torch.nn.functional as F
    # Nearest neighbor interpolation
    # reshaped = rearrange(local_map, 'b (h w) -> b h w', h=32, w=32)
    # upsampled = repeat(reshaped, 'b h w -> b (h 2) (w 2)')
    # result = rearrange(upsampled, 'b h w -> b (h w)')

    # Cubic linear interpolation
    reshaped = rearrange(local_map, 'b (h w) -> b 1 h w', h=32, w=32)
    upsampled = F.interpolate(reshaped, size=(64, 64), mode='bicubic', align_corners=False)
    result = rearrange(upsampled, 'b 1 h w -> b (h w)')
    
    return result

class Scheduler(SchedulerMixin, ConfigMixin):
    order = 1

    temperatures: torch.Tensor

    @register_to_config
    def __init__(
        self,
        mask_token_id: int,
        masking_schedule: str = "cosine",
    ):
        self.temperatures = None
        self.timesteps = None
        # for visualization the mask selected in each step
        self.mask_selected = None
        
    def set_timesteps(
        self,
        num_inference_steps: int,
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
        device: Union[str, torch.device] = None,
    ):
        self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)

        if isinstance(temperature, (tuple, list)):
            self.temperatures = torch.linspace(temperature[0], temperature[1], num_inference_steps, device=device)
        else:
            self.temperatures = torch.linspace(temperature, 0.01, num_inference_steps, device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.long,
        sample: torch.LongTensor,
        starting_mask_ratio: int = 1,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        local_guidance: float = 0,
        local_scores: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
    ) -> Union[SchedulerOutput, Tuple]:
        two_dim_input = sample.ndim == 3 and model_output.ndim == 4

        if two_dim_input:
            batch_size, codebook_size, height, width = model_output.shape
            sample = sample.reshape(batch_size, height * width)
            model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)

        unknown_map = sample == self.config.mask_token_id

        # probs contains the model's predicted probability for all possible tokens at each position
        probs = model_output.softmax(dim=-1)
        probs_ = probs.to(generator.device) if generator is not None else probs
        if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
            probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
        probs_ = probs_.reshape(-1, probs.size(-1))

        # Select the mask position by confidence, pred_original_sample is the specific token ID 
        # obtained by sampling based on these probabilities (all 8912 tokens are predicted)
        pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=probs.device)
        pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
        # For masked positions (unknown_map is True), use the new token obtained by sampling; 
        # for unmasked positions, keep the original token unchanged
        pred_original_sample = torch.where(unknown_map, pred_original_sample, sample) # torch.Size([1, 1024])

        if local_guidance > 0 and local_scores is not None:
            # Tokens where local_map is True (this means they do not need to be flipped, just keep them as they are)
            if unknown_map.shape[-1] == 4096:
                local_scores = upsample_score_map(local_scores)
            local_map = local_scores < local_guidance  # all the map shape is torch.Size([1, 1024])
            joint_map = unknown_map & local_map.to(unknown_map.device) # 这一轮中翻转的，且需要保持为原有的token的位置
            # Flip the area that attention does not focus on back to the original image
            pred_original_sample = torch.where(joint_map, ref_latents.view(ref_latents.shape[0], -1), pred_original_sample)

        if timestep == 0:
            prev_sample = pred_original_sample
        else:
            seq_len = sample.shape[1]
            step_idx = (self.timesteps == timestep).nonzero()
            ratio = (step_idx + 1) / len(self.timesteps)

            if self.config.masking_schedule == "cosine":
                mask_ratio = torch.cos(ratio * math.pi / 2)
            elif self.config.masking_schedule == "linear":
                mask_ratio = 1 - ratio
            else:
                raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

            mask_ratio = starting_mask_ratio * mask_ratio

            mask_len = (seq_len * mask_ratio).floor()  # [bs, 1]
            # Get the probability of predicting the token at each position. 
            # Use torch.gather to extract the probability value corresponding to the selected token from probs
            # This probability value is subsequently used to determine which positions need to be re-masked 
            # (positions with lower confidence are more likely to be masked)
            selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
            # Ignores the tokens given in the input by overwriting their confidence.
            # Maximize the probability of unmasked locations to ensure they are not selected by the mask
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
 
            # do not mask more than amount previously masked
            mask_len = torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            # mask at least one
            mask_len = torch.max(torch.tensor([1], device=model_output.device), mask_len)  # [bs, 1]
            masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)
            self.mask_selected = masking
            
            # Masks tokens with lower confidence.
            prev_sample = torch.where(masking, self.config.mask_token_id, pred_original_sample)

        if two_dim_input:
            prev_sample = prev_sample.reshape(batch_size, height, width)
            pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return SchedulerOutput(prev_sample, pred_original_sample)

    def add_noise(self, sample, timesteps, generator=None):
        step_idx = (self.timesteps == timesteps).nonzero()
        ratio = (step_idx + 1) / len(self.timesteps)

        if self.config.masking_schedule == "cosine":
            mask_ratio = torch.cos(ratio * math.pi / 2)
        elif self.config.masking_schedule == "linear":
            mask_ratio = 1 - ratio
        else:
            raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

        # By generating random values ​​on the current sample and comparing them with mask_ratio, 
        #   we can determine which positions will be masked.
        mask_indices = (
            torch.rand(
                sample.shape, device=generator.device if generator is not None else sample.device, generator=generator
            ).to(sample.device)
            < mask_ratio
        )

        masked_sample = sample.clone()

        # Where mask_indices is True, replace the corresponding value in masked_sample with mask_token_id
        masked_sample[mask_indices] = self.config.mask_token_id

        return masked_sample

