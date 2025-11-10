# Copyright 2025 EditMGT Team. All rights reserved.
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import List
from src.dataset_utils import tokenize_prompt
from termcolor import cprint

def tokens_to_attn(pipe, global_enable=False, prompt:str=None):
    """
    Extracting attention weights of transformer blocks
    """
    attention_weights, global_attention_weights, local_attention_weights = [], [], []

    if pipe.attention_enable_blocks is None:
        return None, None, None
    
    if pipe.local_query_text is not None:
        print('[Enable] use query text for detailed text to image attetion map')
        query_positions = find_query_token_positions(tokenizer=[pipe.tokenizer, pipe.tokenizer_t5] if pipe.tokenizer_t5 is not None else pipe.tokenizer,
                                                    text_encoder_architecture=pipe.transformer.text_encoder_architecture,
                                                    prompt=prompt, query_text=pipe.local_query_text)
    
    for block_idx in pipe.attention_enable_blocks:
        if block_idx < len(pipe.transformer.transformer_blocks):
            processor = pipe.transformer.transformer_blocks[block_idx].attn.processor
        elif block_idx - len(pipe.transformer.transformer_blocks) < len(pipe.transformer.single_transformer_blocks):
            single_idx = block_idx - len(pipe.transformer.transformer_blocks)
            processor = pipe.transformer.single_transformer_blocks[single_idx].attn.processor
        else:
            continue

        # torch.Size([2*bs, head, 1024, 1024]) -> torch.Size([bs, head, 1024, 1024])
        _, attn_weight = processor.attention_weights['text_to_image'].chunk(2)
        attention_weights.append(attn_weight)
        
        _, local_attn_weight = processor.attention_weights['text_to_image'].chunk(2)
        if pipe.local_query_text is not None:
            local_attn_weight = local_attn_weight[:, :, query_positions, :] # torch.Size([bs, heads, text_seq, seq])
        local_attention_weights.append(local_attn_weight)

        if global_enable and processor.attention_weights['reference_text_to_image'] is not None:
            _, global_attn_weight = processor.attention_weights['reference_text_to_image'].chunk(2)
            global_attention_weights.append(global_attn_weight)
    
    if global_enable:
        reference_stacked = torch.stack(global_attention_weights, dim=1)  # [bs, layers, heads, seq, seq] 
    else:  
        reference_stacked = None

    stacked = torch.stack(attention_weights, dim=1)  # [bs, layers, heads, seq, seq]
    local_stacked = torch.stack(local_attention_weights, dim=1)  # [bs, layers, heads, text_seq, seq]
    local_stacked = torch.mean(local_stacked, dim=(1,2,3))  # [bs, seq]
    
    return stacked, local_stacked, reference_stacked  # [bs, layers, heads, text_seq, seq], [bs, seq], [bs, layers, heads, text_seq, seq]


def rescale_scores(scores):
    if scores is None:
        return None
    """Rescale scores to [0, 1] range"""
    scores_min = torch.min(scores, dim=-1, keepdim=True)[0]
    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    return (scores - scores_min) / (scores_max - scores_min + 1e-8)

def find_query_token_positions(tokenizer, text_encoder_architecture, prompt: str, query_text: str) -> List[int]:
    """
    Find the token position corresponding to query_text in prompt
    """ 
    try:
        if isinstance(tokenizer, list):
            working_tokenizer = tokenizer[0]
            working_architecture = 'CLIP'
        else:
            working_tokenizer = tokenizer
            working_architecture = text_encoder_architecture
        
        if isinstance(tokenizer, list):
            full_input_ids = tokenize_prompt(
                working_tokenizer, 
                prompt, 
                'CLIP',
                device=None
            )[0]
            query_input_ids = tokenize_prompt(
                working_tokenizer, 
                query_text, 
                'CLIP',
                device=None
            )[0]
        else:
            full_input_ids = tokenize_prompt(
                working_tokenizer, 
                prompt, 
                working_architecture,
                device=None
            )[0]
            query_input_ids = tokenize_prompt(
                working_tokenizer, 
                query_text, 
                working_architecture,
                device=None
            )[0]
        
        full_tokens = working_tokenizer.convert_ids_to_tokens(full_input_ids)
        query_tokens = working_tokenizer.convert_ids_to_tokens(query_input_ids)
        
        def is_valid_token(token):
            return (token not in ['<|startoftext|>', '<|endoftext|>', '<pad>', '</w>', '!'] 
                    and not token.startswith('<|') 
                    and token.strip() != '')
        
        full_tokens_clean = []
        full_token_positions = []
        for i, token in enumerate(full_tokens):
            if is_valid_token(token):
                full_tokens_clean.append(token.lower().replace('</w>', ''))
                full_token_positions.append(i)
        
        query_tokens_clean = []
        for token in query_tokens:
            if is_valid_token(token):
                query_tokens_clean.append(token.lower().replace('</w>', ''))
        
        print(f"Full tokens (clean): {full_tokens_clean}")
        print(f"Query tokens (clean): {query_tokens_clean}")
        

        if not query_tokens_clean:
            print("No valid query tokens found")
            return []
        
        positions = []
        for i in range(len(full_tokens_clean) - len(query_tokens_clean) + 1):
            match = True
            for j, query_token in enumerate(query_tokens_clean):
                if i + j >= len(full_tokens_clean) or query_token != full_tokens_clean[i + j]:
                    match = False
                    break
            if match:
                positions.extend([full_token_positions[i + j] for j in range(len(query_tokens_clean))])
                print(f"Exact match found: positions {positions}")
                break
        
        positions = sorted(list(set(positions)))
        print(f"Final query token positions: {positions}")
        return positions
        
    except Exception as e:
        print(f"Error finding token positions: {e}")
        import traceback
        traceback.print_exc()
        return []

def smooth_local_scores(local_scores, method='gaussian', strength=1.0, preserve_peaks=True):
    """
    Smooths local scores to make high-value areas more coherent.

    Args:
    local_scores: torch.Tensor, shape [1, 1024] or [1, 32, 32]
    method: str, smoothing method ('gaussian', 'bilateral', 'morphology', 'adaptive')
    strength: float, smoothing strength (0.5-2.0)
    preserve_peaks: bool, whether to preserve the original peaks

    Returns:
    smoothed_scores: torch.Tensor, smoothed scores
    """
    if local_scores is None:
        return None
    # Ensure input is in the format [1, 32, 32]
    if local_scores.shape[-1] == 1024:
        scores_2d = local_scores.reshape(1, 32, 32)
    elif local_scores.shape[-1] == 4096:
        scores_2d = local_scores.reshape(1, 64, 64)
    else:
        scores_2d = local_scores
    
    # Convert to numpy for processing
    scores_np = scores_2d.squeeze(0).cpu().numpy()
    original_scores = scores_np.copy()
    
    if method == 'gaussian':
        # Gaussian filter smoothing
        sigma = strength * 2.0
        smoothed = gaussian_filter(scores_np, sigma=sigma, mode='reflect')
        
    elif method == 'bilateral':
        scores_uint8 = (scores_np * 255).astype(np.uint8)
        d = int(strength * 9)        # Neighborhood diameter
        sigma_color = strength * 75  # sigma value of color space filter
        sigma_space = strength * 75  # sigma value of coordinate space filter
        smoothed_uint8 = cv2.bilateralFilter(scores_uint8, d, sigma_color, sigma_space)
        smoothed = smoothed_uint8.astype(np.float32) / 255.0
        
    elif method == 'morphology':
        from skimage import morphology
        kernel_size = max(3, int(strength * 5))
        kernel = morphology.disk(kernel_size)
        smoothed = morphology.opening(scores_np, kernel)
        smoothed = morphology.closing(smoothed, kernel)
        
    elif method == 'adaptive':
        gaussian_smoothed = gaussian_filter(scores_np, sigma=strength * 1.5, mode='reflect')
        
        local_var = ndimage.generic_filter(scores_np, np.var, size=5)
        local_var = (local_var - local_var.min()) / (local_var.max() - local_var.min() + 1e-8)
        
        adaptive_weight = 1.0 - local_var * 0.7
        smoothed = adaptive_weight * gaussian_smoothed + (1 - adaptive_weight) * scores_np
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if preserve_peaks:
        # Find the original peak point
        peak_threshold = np.percentile(original_scores, 90)
        peak_mask = original_scores > peak_threshold
        
        # Keep more original values ​​at peak positions
        alpha = 0.7
        smoothed[peak_mask] = alpha * original_scores[peak_mask] + (1 - alpha) * smoothed[peak_mask]

    if smoothed.max() > smoothed.min():
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
        original_min, original_max = original_scores.min(), original_scores.max()
        smoothed = smoothed * (original_max - original_min) + original_min

    result = torch.from_numpy(smoothed).unsqueeze(0).to(local_scores.device)
    
    if local_scores.shape[-1] == 1024:
        result = result.reshape(1, 1024)
    elif local_scores.shape[-1] == 4096:
        result = result.reshape(1, 4096)
    
    return result