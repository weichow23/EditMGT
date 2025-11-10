# Copyright 2025 EditMGT Team. All rights reserved.
import argparse
import copy
import logging
import math
import os
import json
from pathlib import Path
import wandb
import sys
sys.path.append(os.getcwd())
import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
import diffusers.optimization
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from PIL import Image
from termcolor import cprint
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import DataLoader, default_collate
from torchvision import transforms
from torchvision.utils import make_grid
from diffusers import EMAModel
from diffusers.loaders import LoraLoaderMixin
from diffusers.utils import is_wandb_available
from src.v2_utils import save_checkpoint, prepare_cond_token
from src.transformer import Transformer2DModel, get_text_encoder_length, load_lora_adapter
from src.pipeline import _prepare_latent_image_ids
from src.dataset_utils import HuggingFaceDataset, HuggingFaceEditDataset, HDFSParquetDataset, HDFSEditParquetDataset, \
                                tokenize_prompt, encode_prompt, move_batch_prompt_device
from src.v2_model import init_text_encoder, init_base_model, negative_prompt, get_pipeline

logger = get_logger(__name__, log_level="INFO")

import torch._dynamo
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NCCL_DEBUG'] = 'ERROR'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_encoder_architecture",
        type=str,
        default="CLIP",
        required=False,
        help="The architecture of the text encoder. One of ['CLIP', 'open_clip', 'flan-t5-base','Qwen2-0.5B','gemini-2b',long_CLIP_T5_base','CLIP_T5_base', 'CLIP_Gemma']",
    )
    parser.add_argument(
        "--instance_dataset",
        type=str,
        default=None,
        required=False,
        help="The dataset to use for training. One of ['MSCOCO600K', 'PickaPicV2']",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    ) # Collov-Labs/Monetico for 512; MeissonFlow/Meissonic for 1024
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="muse_training",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0003,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--validation_prompts", type=str, nargs="*")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--split_vae_encode", type=int, required=False, default=None)
    parser.add_argument("--min_masking_rate", type=float, default=0.0)
    parser.add_argument("--cond_dropout_prob", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", default=50.0, type=float, help="Max gradient norm.", required=False)
    parser.add_argument("--use_lora", action="store_true", help="Fine tune the model using LoRa")
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_target_modules", default=["to_q", "to_k", "to_v"], type=str, nargs="+")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--prompt_prefix", type=str, required=False, default=None)
    parser.add_argument("--wandb_id", default=None)
    parser.add_argument("--train_edit_model", default=False, help='Train T2I model is False; Train edit model is True')
    args = parser.parse_args()

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    return args

def main(args):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        if args.report_to == "wandb":
            wandb.init(project='editmgt', entity=os.environ["WANDB_ENTITY"], name=args.output_dir.split('/')[-1], id=args.wandb_id)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        accelerator.init_trackers("meic", config=vars(copy.deepcopy(args)))

    if args.seed is not None:
        set_seed(args.seed)

    text_encoder, tokenizer, text_att_dim, text_proj_dim = init_text_encoder(text_encoder_architecture=args.text_encoder_architecture,
                                pretrained_model_name_or_path=args.pretrained_model_name_or_path, 
                                revision=args.revision, variant=args.variant, return_dim=True)
    vq_model, scheduler = init_base_model(pretrained_model_name_or_path=args.pretrained_model_name_or_path, revision=args.revision, variant=args.variant)

    if isinstance(text_encoder, list):
        text_encoder[0].eval()
        text_encoder[0].requires_grad_(False)
        text_encoder[1].eval()
        text_encoder[1].requires_grad_(False)
    else:
        text_encoder.eval()
        text_encoder.requires_grad_(False)

    vq_model.requires_grad_(False)

    if args.text_encoder_architecture == 'CLIP':
        model = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, 
                                        subfolder="transformer", low_cpu_mem_usage=False, device_map=None)  
    else:
        model = Transformer2DModel(
            patch_size = 1,
            in_channels = 64,
            num_layers = 14,
            num_single_layers = 28,
            attention_head_dim = 128,
            num_attention_heads = 8,
            joint_attention_dim = text_att_dim,
            pooled_projection_dim = text_proj_dim,
            guidance_embeds = False,
            axes_dims_rope = (16, 56, 56),
            downsample= args.resolution==1024,
            upsample= args.resolution==1024,
            text_encoder_architecture=args.text_encoder_architecture,
            connector_type='linear' if 'CLIP' != args.text_encoder_architecture else None,
        )
        model_tmp = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path,  subfolder="transformer",
            low_cpu_mem_usage=False, device_map=None)
        state_dict = model_tmp.state_dict()

        def load_state_dict_with_skip(model, state_dict):
            model_state_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
            model.load_state_dict(filtered_state_dict, strict=False)

        load_state_dict_with_skip(model, state_dict)

        del model_tmp 

    model = torch.compile(model)

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
        )
        model.add_adapter(lora_config)

    model.train()

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.use_ema:
        ema = EMAModel(
            model.parameters(),
            decay=args.ema_decay,
            update_after_step=args.ema_update_after_step,
            model_cls= Transformer2DModel,
            model_config=model.config,
        )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model_ in models:
                if isinstance(model_, type(accelerator.unwrap_model(model))):
                    if args.use_lora:
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model_)
                    else:
                        model_.save_pretrained(os.path.join(output_dir, "transformer"))
                else:
                    raise ValueError(f"unexpected save model: {model_.__class__}")

                weights.pop()

            if transformer_lora_layers_to_save is not None:
                LoraLoaderMixin.save_lora_weights(
                    output_dir,
                    unet_lora_layers=transformer_lora_layers_to_save,
                    text_encoder_lora_layers=None,
                )
                class LoraConfigEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, set):
                            return list(obj)
                        return super().default(obj)
                with open(os.path.join(output_dir, "lora_config.json"), 'w') as f:
                    json.dump(lora_config.to_dict(), f, indent=2, cls=LoraConfigEncoder)

            if args.use_ema:
                ema.save_pretrained(os.path.join(output_dir, "ema_model"))

    def load_model_hook(models, input_dir):
        transformer = None

        def adap_compile(ori_dict):
            new_dict = {}
            for k,v in ori_dict.items():
                new_dict['_orig_mod.'+k] = v
            return new_dict

        while len(models) > 0:
            model_ = models.pop()
            if isinstance(model_, type(accelerator.unwrap_model(model))):
                if args.use_lora:
                    transformer = model_
                else:
                    load_model = Transformer2DModel.from_pretrained(os.path.join(input_dir, "transformer"),low_cpu_mem_usage=False,device_map=None)
                    model_.load_state_dict(adap_compile(load_model.state_dict()))
                    del load_model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer is not None:
            transformer = load_lora_adapter(transformer, input_dir)

        if args.use_ema:
            ema_model_path = os.path.join(input_dir, "ema_model")
            # If you skip loading EMA, it is equivalent to starting a new EMA. 
            # It is used for training that was not EMA before and switching to EMA training.
            if os.path.exists(ema_model_path) and os.path.isdir(ema_model_path):
                try:
                    load_from = EMAModel.from_pretrained(ema_model_path, model_cls=Transformer2DModel)
                    ema.load_state_dict(adap_compile(load_from.state_dict()))
                    del load_from
                    cprint("EMA model loaded successfully", 'red')
                except Exception as e:
                    cprint(f"Failed to load EMA model: {e}. Continuing without EMA state.", 'red')
            else:
                cprint(f"EMA model not found at {ema_model_path}. Continuing without EMA state.", 'red')

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.adam_weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optimizer_cls(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    logger.info("Creating dataloaders and lr_scheduler")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable_params = sum(p.numel() for p in trainable_params)
    cprint(f"Number of Model parameters: {sum(p.numel() for p in model.parameters()) // 1024 // 1024} M", 'cyan')
    cprint(f"Number of trainable parameters: {num_trainable_params//1024//1024} M", 'cyan')

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if args.instance_dataset == "HDFSParquetDataset":
        dataset = HDFSParquetDataset(
            root_dir=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
    elif args.instance_dataset == "HDFSEditParquetDataset":
        dataset = HDFSEditParquetDataset(
            root_dir=args.instance_data_dir,
            tokenizer=tokenizer,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
    elif args.instance_dataset == 'HuggingFaceDataset':
        dataset = HuggingFaceDataset(
            hf_dataset=load_dataset(args.instance_data_dir, split="train"),
            tokenizer=tokenizer,
            image_key='image_file',
            prompt_key='edit_instruction',
            prompt_prefix=args.prompt_prefix,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
        )
    elif args.instance_dataset == 'HuggingFaceEditDataset':
        dataset = HuggingFaceEditDataset(
            hf_dataset=load_dataset(args.instance_data_dir, split="replace"), # example: read Bin1117/AnyEdit
            tokenizer=tokenizer,
            image_key='image_file',
            prompt_key='edit_instruction',
            prompt_prefix=args.prompt_prefix,
            size=args.resolution,
            text_encoder_architecture=args.text_encoder_architecture,
        )
    else:
        assert False

    if args.instance_dataset in ["HDFSParquetDataset", 'HDFSEditParquetDataset']:
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            num_workers=accelerator.num_processes,
            collate_fn=default_collate,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            collate_fn=default_collate,
            pin_memory=True,
        )

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    )

    logger.info("Preparing model, optimizer and dataloaders")

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader
    )

    train_dataloader.num_batches = len(train_dataloader)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if isinstance(text_encoder, list):
        text_encoder[0].to(device=accelerator.device, dtype=weight_dtype)
        text_encoder[1].to(device=accelerator.device, dtype=weight_dtype)
    else:
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    vq_model.to(device=accelerator.device)

    if args.use_ema:
        ema.to(accelerator.device)

    with torch.no_grad():
        empty_embeds, empty_clip_embeds = encode_prompt(
            text_encoder, tokenize_prompt(tokenizer, "", args.text_encoder_architecture, device=accelerator.device, non_blocking=True), args.text_encoder_architecture
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            if len(dirs) > 0:
                resume_from_checkpoint = os.path.join(args.output_dir, dirs[-1])
            else:
                resume_from_checkpoint = None

        if resume_from_checkpoint is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
        else:
            cprint('resume successfully!', 'red')
            accelerator.print(f"Resuming from checkpoint {resume_from_checkpoint}")

    if resume_from_checkpoint is None:
        global_step = 0
        first_epoch = 0
    else:
        accelerator.load_state(resume_from_checkpoint)  # load opt and model
        global_step = int(os.path.basename(resume_from_checkpoint).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch

    # This is to solve the inconsistent tensor device issue
    if args.use_ema:
        ema.shadow_params = [p.to(accelerator.device) for p in ema.shadow_params]
    
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            torch.cuda.empty_cache()
            with torch.no_grad():
                micro_conds = batch["micro_conds"].to(accelerator.device, non_blocking=True)
                pixel_values = batch["image"].to(accelerator.device, non_blocking=True)

                image_tokens = prepare_cond_token(split_vae_encode=args.split_vae_encode, 
                                        pixel_values=pixel_values, vq_model=vq_model) # 512: [bs, 1024]

                batch_size, seq_len = image_tokens.shape

                timesteps = torch.rand(batch_size, device=image_tokens.device)
                mask_prob = torch.cos(timesteps * math.pi * 0.5)
                mask_prob = mask_prob.clip(args.min_masking_rate)

                num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
                batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
                mask = batch_randperm < num_token_masked.unsqueeze(-1)

                mask_id = accelerator.unwrap_model(model).config.vocab_size - 1
                input_ids = torch.where(mask, mask_id, image_tokens)
                labels = torch.where(mask, image_tokens, -100)

                with torch.no_grad():
                    batch["prompt_input_ids"] = move_batch_prompt_device(prompts=batch["prompt_input_ids"], device=accelerator.device, 
                                                                        split=isinstance(text_encoder, list), non_blocking=True)
                    encoder_hidden_states, cond_embeds = encode_prompt(
                            text_encoder, batch["prompt_input_ids"], args.text_encoder_architecture
                    )

                if args.cond_dropout_prob > 0.0:
                    assert encoder_hidden_states is not None

                    batch_size = encoder_hidden_states.shape[0]

                    mask = (
                        torch.zeros((batch_size, 1, 1), device=encoder_hidden_states.device).float().uniform_(0, 1)
                        < args.cond_dropout_prob
                    )

                    empty_embeds_ = empty_embeds.expand(batch_size, -1, -1)
                    encoder_hidden_states = torch.where(
                        (encoder_hidden_states * mask).bool(), encoder_hidden_states, empty_embeds_
                    )

                    empty_clip_embeds_ = empty_clip_embeds.expand(batch_size, -1)
                    cond_embeds = torch.where((cond_embeds * mask.squeeze(-1)).bool(), cond_embeds, empty_clip_embeds_)

                bs = input_ids.shape[0]
                vae_scale_factor = 2 ** (len(vq_model.config.block_out_channels) - 1)
                resolution = args.resolution // vae_scale_factor
                input_ids = input_ids.reshape(bs, resolution, resolution)

                # --- edit
                if args.train_edit_model:
                    reference_values = batch["reference_image"].to(accelerator.device, non_blocking=True)
                    reference_image_hidden_states = prepare_cond_token(split_vae_encode=args.split_vae_encode, pixel_values=reference_values, vq_model=vq_model)
                    reference_image_hidden_states = reference_image_hidden_states.reshape(bs, resolution, resolution)
                # ----

            with torch.no_grad():
                batch["prompt_input_ids"] = move_batch_prompt_device(prompts=batch["prompt_input_ids"], device=accelerator.device, 
                                                                    split=isinstance(text_encoder, list), non_blocking=True)
                encoder_hidden_states, cond_embeds = encode_prompt(
                    text_encoder, batch["prompt_input_ids"], args.text_encoder_architecture
                )

            # Train Step
            with accelerator.accumulate(model):
                codebook_size = accelerator.unwrap_model(model).config.codebook_size
                img_ids = _prepare_latent_image_ids(input_ids.shape[-2], input_ids.shape[-1],
                                                    input_ids.device, input_ids.dtype, up_sample=args.resolution!=1024) # torch.Size([1024, 3])

                if not isinstance(text_encoder, list):
                    txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(device = input_ids.device, dtype = input_ids.dtype)
                else:
                    txt_ids = torch.zeros(get_text_encoder_length(args.text_encoder_architecture, return_main=True),3).to(device = input_ids.device, dtype = input_ids.dtype)

                if args.train_edit_model:
                    reference_image_ids = _prepare_latent_image_ids(reference_image_hidden_states.shape[-2],
                                                        reference_image_hidden_states.shape[-1],
                                                        reference_image_hidden_states.device,
                                                        reference_image_hidden_states.dtype,
                                                        up_sample= args.resolution!=1024)

                logits = (
                    model(
                        hidden_states=input_ids, # should be (batch size, channel, height, width)
                        encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                        micro_conds=micro_conds, # 
                        pooled_projections=cond_embeds, # should be (batch_size, projection_dim)
                        img_ids = img_ids,
                        txt_ids = txt_ids,
                        timestep = mask_prob * 1000,
                        # ----- edit
                        **({
                            'reference_image_hidden_states': reference_image_hidden_states,
                            'reference_image_ids': reference_image_ids,
                            'lora_part_enable': True
                        } if args.train_edit_model else {})
                    )
                    .reshape(bs, codebook_size, -1)
                    .permute(0, 2, 1)
                    .reshape(-1, codebook_size)
                )

                loss = F.cross_entropy(
                    logits,
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_masking_rate = accelerator.gather(mask_prob.repeat(args.train_batch_size)).mean()

                accelerator.backward(loss)

                if args.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema.step(model.parameters())

                if (global_step + 1) % args.logging_steps == 0:
                    logs = {
                        "step_loss": avg_loss.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss: {avg_loss.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                if (global_step + 1) % args.checkpointing_steps == 0:
                    save_checkpoint(args, accelerator, global_step + 1, logger)

                if (global_step + 1) % args.validation_steps == 0 and accelerator.is_main_process:
                    if args.use_ema:
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())

                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        logger.info("Generating images...")

                        model.eval()
                        pipe = get_pipeline(text_encoder_architecture=args.text_encoder_architecture,
                                            transformer=accelerator.unwrap_model(model),
                                            tokenizer=tokenizer,
                                            text_encoder=text_encoder,
                                            vq_model=vq_model,
                                            scheduler=scheduler,
                                        )
                        if args.train_edit_model:
                            validation_prompts = ['A woman with short hair wears a silver gas mask.', 
                                                  'wear a sunglasses on the dog', 
                                                  "A woman wearing a white suspender skirt is sitting"]
                            validation_images = ['_Rh_zxIUWXA.jpg', '0eKR4M2uuL8.jpg', '__Owak0IgJk.jpg']
                            pil_images = pipe(prompt=validation_prompts,
                                negative_prompt=[negative_prompt] * len(validation_prompts),
                                height=args.resolution, 
                                width=args.resolution,
                                reference_image=[Image.open(f'assets/inpaint/{img}') for img in validation_images],
                                reference_strength=1,
                                guidance_scale=9, num_inference_steps=32).images
                        else:
                            validation_prompts = args.validation_prompts
                            pil_images = pipe(prompt=validation_prompts,
                                height=args.resolution,
                                width=args.resolution,
                                guidance_scale=9,
                                num_inference_steps=32).images
                            
                        result=[]
                        for img in pil_images:
                            if not isinstance(img, torch.Tensor):
                                img = transforms.ToTensor()(img)
                            result.append(img.unsqueeze(0))
                        result = torch.cat(result,dim=0)
                        result = make_grid(result, nrow=3)
                        # save_image(result,os.path.join(args.output_dir,str(global_step)+'_text2image_512_CFG-9.png'))

                        wandb_images = [
                            wandb.Image(image, caption=validation_prompts[i])
                            for i, image in enumerate(pil_images)
                        ]
                        wandb.log({"generated_images": wandb_images}, step=global_step + 1)

                        model.train()

                    if args.use_ema:
                        ema.restore(model.parameters())

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= args.max_train_steps:
                break

            if accelerator.sync_gradients:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(args, accelerator, global_step, logger)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            ema.copy_to(model.parameters())
        model.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main(parse_args())
