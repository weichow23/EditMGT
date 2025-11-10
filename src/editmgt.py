# Copyright 2025 EditMGT Team. All rights reserved.
import torch
from diffusers import VQModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, AutoModelForCausalLM, AutoTokenizer
from src.scheduler import Scheduler
from src.pipeline import Pipeline
from src.transformer import Transformer2DModel

def init_edit_mgt(device, enable_bf16=False):
    base_model_path = 'WeiChow/EditMGT'
    text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
        base_model_path, subfolder="text_encoder"
    )
    tokenizer_clip = CLIPTokenizer.from_pretrained(
        base_model_path, subfolder="tokenizer"
    )
    model_gemma = AutoModelForCausalLM.from_pretrained(base_model_path, subfolder="llm_encoder", output_hidden_states=True)
    tokenizer_gemma = AutoTokenizer.from_pretrained(base_model_path, subfolder="llm_encoder")
    text_encoder = [text_encoder_clip,model_gemma]
    tokenizer = [tokenizer_clip,tokenizer_gemma]
    vq_model = VQModel.from_pretrained(
        base_model_path, subfolder="vqvae",
    ) 
    scheduler = Scheduler.from_pretrained(
        base_model_path, subfolder="scheduler",
    )
    if enable_bf16:
        text_encoder = [t.to(torch.bfloat16) for t in text_encoder]
        vq_model = vq_model.to(torch.bfloat16)
        # model = Transformer2DModel.from_pretrained(base_model_path, subfolder="transformer", torch_dtype=torch.bfloat16)
        model = Transformer2DModel.from_pretrained(base_model_path, subfolder="editmgt", torch_dtype=torch.bfloat16)
    else:
        # model = Transformer2DModel.from_pretrained(base_model_path, subfolder="transformer")
        model = Transformer2DModel.from_pretrained(base_model_path, subfolder="editmgt")
    
    pipe = Pipeline(
        transformer=model,
        tokenizer=tokenizer[0],
        text_encoder=text_encoder[0],
        vqvae=vq_model,
        scheduler=scheduler,
        text_encoder_t5=text_encoder[1],
        tokenizer_t5=tokenizer[1]
    )
    pipe = pipe.to(device)

    return pipe