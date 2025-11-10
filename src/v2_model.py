# Copyright 2025 EditMGT Team. All rights reserved.
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, AutoModel, AutoTokenizer
from diffusers import VQModel
from src.scheduler import Scheduler
from src.pipeline import Pipeline
from src.transformer import get_text_encoder_length
from termcolor import cprint

negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"


def init_base_model(pretrained_model_name_or_path, revision=None, variant=None):
    vq_model = VQModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="vqvae", revision=revision, variant=variant
    ) 
    scheduler = Scheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler", revision=revision, variant=variant,
    )
    return vq_model, scheduler

def test_text_encoder(tokenizer, text_enc):
    text = "meissonic"
    inputs = tokenizer(text, return_tensors="pt")
    print(len(inputs))
    with torch.no_grad():
        outputs = text_enc(**inputs, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states
    for i, layer_embedding in enumerate(hidden_states):
        print(f"Layer {i} embedding shape: {layer_embedding.shape}")

def init_text_encoder(text_encoder_architecture, pretrained_model_name_or_path=None, revision=None, variant=None, return_dim=False):
    if text_encoder_architecture == "CLIP":
        text_encoder = CLIPTextModelWithProjection.from_pretrained( 
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, variant=variant
        )
        joint_attention_dim, pooled_projection_dim = 1024, 1024
    elif text_encoder_architecture == "open_clip":
        text_encoder = CLIPTextModelWithProjection.from_pretrained("apple/DFN5B-CLIP-ViT-H-14-384", subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained("apple/DFN5B-CLIP-ViT-H-14-384")
        joint_attention_dim, pooled_projection_dim = 1024, 1024

    elif text_encoder_architecture in ["CLIP_T5-l", "CLIP_T5-xl", "CLIP_T5-xxl"]:
        text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, variant=variant
        )
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        if text_encoder_architecture == "CLIP_T5-xl":
            text_encoder_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
            tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-xl")
            joint_attention_dim, pooled_projection_dim = 2048, 1024
        elif text_encoder_architecture == "CLIP_T5-xxl":
            text_encoder_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
            tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
            joint_attention_dim, pooled_projection_dim = 4096, 1024
        elif text_encoder_architecture == "CLIP_T5-l":
            text_encoder_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
            tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-large")
            joint_attention_dim, pooled_projection_dim = 1024, 1024
        text_encoder = [text_encoder_clip,text_encoder_t5]
        tokenizer = [tokenizer_clip,tokenizer_t5]
    elif text_encoder_architecture == "CLIP_Qwen2.5":
        text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, variant=variant
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer
        qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        text_encoder = [text_encoder_clip,qwen_model]
        tokenizer = [tokenizer_clip,qwen_tokenizer]
        joint_attention_dim, pooled_projection_dim = 896, 1024

    elif text_encoder_architecture == "CLIP_llama":
        text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, variant=variant
        )

        from transformers import AutoModelForCausalLM, AutoTokenizer
        qwen_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        qwen_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        qwen_tokenizer.pad_token_id = qwen_tokenizer.eos_token_id

        text_encoder = [text_encoder_clip,qwen_model]
        tokenizer = [tokenizer_clip,qwen_tokenizer]

        joint_attention_dim, pooled_projection_dim = 2048, 1024

    elif text_encoder_architecture == "Gemma":
        from transformers import AutoModel, AutoTokenizer
        model_name = "google/gemma-2-2b-it"
        text_encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        joint_attention_dim, pooled_projection_dim = 2304, 1024

    elif text_encoder_architecture in ["CLIP_Gemma2", "CLIP_Gemma2-raw", "CLIP_Gemma1"]:
        text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, variant=variant
        )

        joint_attention_dim, pooled_projection_dim = 2304, 1024
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        if text_encoder_architecture == "CLIP_Gemma2":
            model_gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", output_hidden_states=True)
            tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        elif text_encoder_architecture == "CLIP_Gemma1":
            model_gemma = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it", output_hidden_states=True)
            tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
            joint_attention_dim, pooled_projection_dim = 2048, 1024

        else:
            model_gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", output_hidden_states=True)
            tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-2b")

        text_encoder = [text_encoder_clip,model_gemma]
        tokenizer = [tokenizer_clip,tokenizer_gemma]
    
    elif text_encoder_architecture in ["CLIP_Gemma3", "CLIP_Gemma3-raw"]:
        text_encoder_clip = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=revision, variant=variant
        )
        from transformers import AutoModel, AutoTokenizer, Gemma3ForCausalLM
        if text_encoder_architecture == "CLIP_Gemma3":
            model_gemma = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16, output_hidden_states=True)
            tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        else:
            model_gemma = Gemma3ForCausalLM.from_pretrained("google/gemma-3-1b-pt", torch_dtype=torch.bfloat16, output_hidden_states=True)
            tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
        text_encoder = [text_encoder_clip,model_gemma]
        tokenizer = [tokenizer_clip,tokenizer_gemma]

        joint_attention_dim, pooled_projection_dim = 1152, 1024
    else:
        raise ValueError(f"Unknown text encoder architecture: {text_encoder_architecture}")

    if return_dim:
        return text_encoder, tokenizer, joint_attention_dim, pooled_projection_dim
    else:
        return text_encoder, tokenizer

def get_pipeline(
        text_encoder_architecture, 
        transformer,
        tokenizer,
        text_encoder,
        vq_model,
        scheduler,
    ):
    
    if text_encoder_architecture == "CLIP" or text_encoder_architecture == "open_clip" or text_encoder_architecture == "Gemma":
        pipe = Pipeline(
            transformer=transformer,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vqvae=vq_model,
            scheduler=scheduler,
            text_encoder_t5=None,
            tokenizer_t5=None
        )
    else:
        pipe = Pipeline(
            transformer=transformer,
            tokenizer=tokenizer[0],
            text_encoder=text_encoder[0],
            vqvae=vq_model,
            scheduler=scheduler,
            text_encoder_t5=text_encoder[1],
            tokenizer_t5=tokenizer[1]
        )
    return pipe

