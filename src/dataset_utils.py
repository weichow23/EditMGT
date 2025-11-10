# Copyright 2025 EditMGT Team. All rights reserved.
import os
from termcolor import cprint
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL.ImageOps import exif_transpose
import pyarrow.parquet as pq
import yaml
from torch.utils.data import IterableDataset
import pyarrow.fs as fs
import io
import json

def move_batch_prompt_device(prompts, device, split=False, non_blocking=None):
    def move(tensor):
        if non_blocking is not None:
            tensor.to(non_blocking=non_blocking)
        return tensor.to(device=device)
    if split:
        prompts[0] = move(prompts[0])
        prompts[1] = move(prompts[1])
        return prompts
    else:
        return move(prompts)

@torch.no_grad()
def tokenize_prompt(tokenizer, prompt, text_encoder_architecture='CLIP', device=None, non_blocking=None):
    def move_to_device(tensor):
        if non_blocking is not None:
            tensor.to(non_blocking=non_blocking)
        return tensor.to(device=device) if device is not None else tensor
    
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        input_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).input_ids
        return move_to_device(input_ids)
    elif text_encoder_architecture == 'Gemma':
        input_ids = tokenizer(prompt, 
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).input_ids
        return move_to_device(input_ids)
    elif isinstance(tokenizer, list):
        input_ids = []
        input_ids.append(move_to_device(tokenizer[0](
            prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).input_ids))
        input_ids.append(move_to_device(tokenizer[1](
            prompt,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).input_ids))
        return input_ids
    elif text_encoder_architecture == 'flan-t5-base':
        input_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).input_ids
        return move_to_device(input_ids)
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")

def get_encode_hidden_state_len(text_encoder_architecture):
    if text_encoder_architecture == "CLIP":
        return 77
    elif text_encoder_architecture == 'flan-t5-base':
        return 512
    else:
        return 256  # default

def encode_prompt(text_encoder, input_ids, text_encoder_architecture='CLIP'):
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        outputs = text_encoder(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2] # [2，77，1024]
        cond_embeds = outputs[0] # [2，1280]

        return encoder_hidden_states, cond_embeds
    elif text_encoder_architecture == 'Gemma':
        outputs = text_encoder(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        cond_embeds = outputs[0]
        print("encoder_hidden_states",encoder_hidden_states.shape) 
        print("cond_embeds",cond_embeds.shape) 
        return encoder_hidden_states, torch.mean(cond_embeds, dim=1) #output shape will be [bs, token_length, 2304]
    elif isinstance(text_encoder, list):
        if "T5" in text_encoder_architecture:
            outputs_clip = text_encoder[0](input_ids=input_ids[0], return_dict=True, output_hidden_states=True)
            cond_embeds_clip = outputs_clip[0]

            outputs_t5 = text_encoder[1](input_ids=input_ids[1], decoder_input_ids=input_ids[1],
                                return_dict=True, output_hidden_states=True)
            encoder_hidden_states_t5 = outputs_t5.encoder_hidden_states[-2]
            return encoder_hidden_states_t5, cond_embeds_clip
        else:
            outputs_clip = text_encoder[0](input_ids=input_ids[0], return_dict=True, output_hidden_states=True)
            encoder_hidden_states_clip = outputs_clip.hidden_states[-2]
            cond_embeds_clip = outputs_clip[0]   # cond_embeds_clip torch.Size([bs, 1024])
            outputs_gemma = text_encoder[1](input_ids=input_ids[1], return_dict=True, output_hidden_states=True)
            encoder_hidden_states_gemma = outputs_gemma.hidden_states[-2]
            cond_embeds_gemma = outputs_gemma[0]
            return encoder_hidden_states_gemma, cond_embeds_clip
    elif text_encoder_architecture == 'flan-t5-base': # To be finished, has bug
        outputs = text_encoder(input_ids=input_ids, decoder_input_ids=input_ids,
                               return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.encoder_hidden_states[-2]
        cond_embeds = outputs[0] # To be finished, has bug here, because t5 does not have global representation embedding
        return encoder_hidden_states, cond_embeds
    elif text_encoder_architecture == 'CLIP_T5_base':
        outputs_clip = text_encoder[0](input_ids=input_ids[0], return_dict=True, output_hidden_states=True)
        outputs_t5 = text_encoder[1](input_ids=input_ids[1], decoder_input_ids=torch.zeros_like(input_ids[1]),
                               return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs_t5.encoder_hidden_states[-2]
        cond_embeds = outputs_clip[0]
        return encoder_hidden_states, cond_embeds
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")


def process_image(image, size, norm=False):
    image = exif_transpose(image)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    orig_height = image.height
    orig_width = image.width

    image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(image)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(size, size))
    image = transforms.functional.crop(image, c_top, c_left, size, size)
    image = transforms.ToTensor()(image)

    if norm:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)

    micro_conds = torch.tensor(
        [orig_width, orig_height, c_top, c_left, 6.0], # [size[0], size[1], 0, 0, 6.0],
    )

    return {"image": image, "micro_conds": micro_conds}


class HDFSParquetDataset(IterableDataset):
    def __init__(self, root_dir, tokenizer=None, 
                 size=512, text_encoder_architecture='CLIP', norm=False,
                 rank=0, world_size=1, shuffle=True, repeat=True):
        super().__init__()
        # Use different seed for each rank to ensure diversity
        random.seed(23 + rank)
        
        self.root_dir = root_dir
        self.yaml_file = './train/train_v2.yaml'
        self.tokenizer = tokenizer
        self.size = size
        self.text_encoder_architecture = text_encoder_architecture
        self.norm = norm
        
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        
        self.hdfs = fs.HadoopFileSystem(host="harunava", port=8020)
        self._init_parquet_file_list()
        
        if self.world_size > 1:
            self._distribute_files_across_ranks()
            print(f"Total {self.world_size} Rank {self.rank} loaded {len(self.parquet_files)} files")

    def __len__(self):
        return 10000000

    def _init_parquet_file_list(self):
        with open(self.yaml_file, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        self.parquet_files = []
        for key, value in yaml_data.items():
            if isinstance(value, dict) and 'ratio' in value:
                hdfs_path = os.path.join(self.root_dir, key)
                all_files = []
                import subprocess
                result = subprocess.run(
                    ["hdfs", "dfs", "-ls", hdfs_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                files = result.stdout.split('\n')
                for line in files:
                    if line.strip():
                        if 'hdfs://' in line:
                            full_path = line[line.index('hdfs://harunavaali'):].replace('hdfs://harunavaali', '')
                            all_files.append(full_path)
                print(key)
                cprint(len(all_files), 'cyan')

                sample_count = int(len(all_files) * value['ratio'])
                if sample_count <= len(all_files):
                    # ratio <= 1, normal sampling (no repetition)
                    sampled_files = random.sample(all_files, k=sample_count)
                else:
                    # ratio > 1, allows repeated sampling
                    sampled_files = random.choices(all_files, k=sample_count)
                self.parquet_files.extend(sampled_files)
                
            elif isinstance(value, list):
                if self.root_dir:
                    full_paths = [os.path.join(self.root_dir, path) if not path.startswith('/') else path 
                                for path in value]
                else:
                    full_paths = value

                sample_count = len(full_paths)
                if sample_count > 0:
                    sampled_files = random.sample(full_paths, k=sample_count)
                else:
                    sampled_files = full_paths
                
                self.parquet_files.extend(sampled_files)
            else:
                print(f"Unsupported format for key {key} in YAML file, skipping")

    def _distribute_files_across_ranks(self):
        """Ensure that different ranks obtain different subsets of files"""
        sorted_files = sorted(self.parquet_files)
        
        # Use a fixed seed to scramble the file to ensure that the scrambling results of all ranks are consistent
        global_random = random.Random(42)
        global_random.shuffle(sorted_files)
        
        # Distribute files evenly to each rank
        rank_files = []
        for i, file_path in enumerate(sorted_files):
            if i % self.world_size == self.rank:
                rank_files.append(file_path)
        
        # Use the rank-specific random seed to shuffle the files of that rank again
        if self.shuffle:
            random.shuffle(rank_files)
            
        self.parquet_files = rank_files

    def _sort_and_shuffle(self, data, seed=42):
        """Sort and shuffle the file list"""
        data.sort()
        # Use rank-specific seeds to ensure different ranks have different orders
        random.Random(seed + self.rank).shuffle(data)
        return data

    def _split_shard(self, data, shard_idx, shard_size):
        """Shard the data"""
        num = len(data)
        if num < shard_size:
            print(f"Data size ({num}) < shard size ({shard_size}), may cause uneven distribution")
            return data if shard_idx == 0 else []
        
        start_idx = (num * shard_idx) // shard_size
        end_idx = (num * (shard_idx + 1)) // shard_size
        return data[start_idx:end_idx]

    def _process_sample(self, sample):
        if not sample:
            print("Sample missing")
            return None
        
        if 'task2' in sample.keys():
            task2_data = sample.get('task2', '{"Caption": "This is a test image"}')
            if isinstance(task2_data, str):
                caption_data = json.loads(task2_data)
            else:
                # Sometimes task2 may already be a dictionary
                caption_data = task2_data
            
            generated_caption = caption_data.get("Caption", None) or caption_data.get("caption", None)
        else:
            generated_caption = sample.get('caption', "This is a test image")
        
        if not generated_caption:
            generated_caption = 'This is a test image'
        
        if 'image' in sample.keys():
            image_path = sample['image']
        else:
            image_path = sample['img']
        
        # Check image data
        if 'bytes' not in image_path:
            print(f"Image data missing 'bytes' field: {image_path.keys() if isinstance(image_path, dict) else 'not a dict'}")
            return None
            
        instance_image = Image.open(io.BytesIO(image_path['bytes']))
        rv = process_image(instance_image, self.size, self.norm)

        if isinstance(self.tokenizer, list):
            _tmp_ = tokenize_prompt(self.tokenizer, generated_caption, self.text_encoder_architecture)
            rv["prompt_input_ids"] = [_tmp_[0][0], _tmp_[1][0]]
        else:
            rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, generated_caption, self.text_encoder_architecture)[0]
        
        return rv

    def _generate_samples(self, seed=42):
        """The main logic of generating samples"""
        # Note: Files are already assigned to each rank during initialization; no further assignment is required.
        # Operate only on files in the current rank.
        worker_files = self.parquet_files.copy()
        
        # Shard by worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if len(worker_files) % worker_info.num_workers != 0:
                print(
                    f"[DATA]--current rank [{self.rank}] file num {len(worker_files)} "
                    f"cannot evenly split to worker_num {worker_info.num_workers}"
                )
            worker_files = self._split_shard(
                worker_files,
                worker_info.id,
                worker_info.num_workers,
            )
        
        if self.shuffle:
            current_seed = seed + self.rank
            if worker_info is not None:
                current_seed += worker_info.id
            random.seed(current_seed)
            random.shuffle(worker_files)
        
        print(f"Rank {self.rank} {'Worker ' + str(worker_info.id) if worker_info else ''} processing {len(worker_files)} files")
        
        # Traverse each file and generate samples
        while True:
            for file_path in worker_files:
                try:
                    table = pq.read_table(file_path, filesystem=self.hdfs)
                    batch_dict = table.to_pydict()
                    
                   # Check if there is data
                    if not batch_dict or all(len(v) == 0 for v in batch_dict.values()):
                        print(f"Empty data in file: {file_path}")
                        continue
                    
                    # Process each line
                    num_rows = len(next(iter(batch_dict.values())))
                    processed_count = 0
                    
                    # Randomly shuffle the order of rows to ensure that different ranks do not read data in the same order
                    row_indices = list(range(num_rows))
                    if self.shuffle:
                        random.shuffle(row_indices)
                    
                    for i in row_indices:
                        sample = {k: v[i] for k, v in batch_dict.items()}
                        processed_sample = self._process_sample(sample)
                        if processed_sample is not None:
                            processed_count += 1
                            yield processed_sample
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            if not self.repeat:
                break
            # Reshuffle the file order at the end of each epoch
            if self.shuffle:
                random.shuffle(worker_files)

    def __iter__(self):
        """Returns an iterator"""
        # Use rank-specific seeds to ensure different ranks have different random sequences
        return self._generate_samples(seed=42 + self.rank)

class HuggingFaceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_key,
        prompt_key,
        prompt_prefix=None,
        size=512,
        text_encoder_architecture='CLIP', 
    ):
        self.size = size
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.tokenizer = tokenizer
        self.hf_dataset = hf_dataset
        self.prompt_prefix = prompt_prefix
        self.text_encoder_architecture = text_encoder_architecture

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]

        rv = process_image(item[self.image_key], self.size)

        prompt = item[self.prompt_key]

        if self.prompt_prefix is not None:
            prompt = self.prompt_prefix + prompt

        if isinstance(self.tokenizer, list): 
            _tmp_ = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)
            rv["prompt_input_ids"] = [_tmp_[0][0],_tmp_[1][0]]
        else:
            rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)[0]

        return rv

class HuggingFaceEditDataset(HuggingFaceDataset):
    def __getitem__(self, index):
        item = self.hf_dataset[index]

        rv = process_image(item['edited_file'], self.size)  # edit results

        prompt = item[self.prompt_key]

        if self.prompt_prefix is not None:
            prompt = self.prompt_prefix + prompt

        if isinstance(self.tokenizer, list): 
            _tmp_ = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)
            rv["prompt_input_ids"] = [_tmp_[0][0],_tmp_[1][0]]
        else:
            rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt, self.text_encoder_architecture)[0]

        rv['reference_image'] = process_image(item["image_file"], size=self.size)['image'] # raw image

        return rv

class HDFSEditParquetDataset(HDFSParquetDataset):
    def __init__(self, root_dir, tokenizer=None, 
                 size=512, text_encoder_architecture='CLIP', norm=False,
                 rank=0, world_size=1, shuffle=True, repeat=True, train_edit_model=True):
        # Use different seeds to ensure diversity in each rank
        random.seed(23 + rank)
        
        self.root_dir = root_dir
        # Force the use of the YAML configuration file of the editing model
        self.yaml_file = './train/train_edit.yaml'
        self.tokenizer = tokenizer
        self.size = size
        self.text_encoder_architecture = text_encoder_architecture
        self.norm = norm
        
        # Distributed training parameters
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        
        self.hdfs = fs.HadoopFileSystem(host="harunava", port=8020)
        self._init_parquet_file_list()
        
        if self.world_size > 1:
            self._distribute_files_across_ranks()
            print(f"Total {self.world_size} Rank {self.rank} loaded {len(self.parquet_files)} files")

    def _process_sample(self, sample):
        if not sample:
            print("Sample missing")
            return None
        
        if 'edit_instruction' in sample.keys():
            instruction = sample['edit_instruction']
        else:
            instruction = sample['instruction']
   
        if 'output' in sample.keys():
            edited_file = Image.open(io.BytesIO(sample['output']['bytes']))
        elif 'output_img' in sample.keys():
            edited_file = Image.open(io.BytesIO(sample['output_img']['bytes']))
        elif 'generated_image' in sample.keys():
            edited_file = Image.open(io.BytesIO(sample['generated_image']))
        else:
            edited_file = Image.open(io.BytesIO(sample['edited_file']['bytes']))

        rv = process_image(edited_file, self.size, self.norm) # edit results

        if isinstance(self.tokenizer, list):
            _tmp_ = tokenize_prompt(self.tokenizer, instruction, self.text_encoder_architecture)
            rv["prompt_input_ids"] = [_tmp_[0][0], _tmp_[1][0]]
        else:
            rv["prompt_input_ids"] = tokenize_prompt(self.tokenizer, instruction, self.text_encoder_architecture)[0]
        
        if 'input' in sample.keys():
            image_file = Image.open(io.BytesIO(sample['input']['bytes']))
        elif 'input_img' in sample.keys():
            image_file = Image.open(io.BytesIO(sample['input_img']['bytes']))
        else:
            try:
                image_file = Image.open(io.BytesIO(sample['image_file']['bytes']))
            except:
                image_file = Image.open(io.BytesIO(sample['image_file']))

        rv['reference_image'] = process_image(image_file, size=self.size, norm=self.norm)['image'] # raw image
        return rv