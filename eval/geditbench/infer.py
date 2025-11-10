import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from src.editmgt import init_edit_mgt
from src.v2_model import negative_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_name", type=str, default="edit_images", help="path to edited images")
    args = parser.parse_args()

    dataset = load_dataset("stepfun-ai/GEdit-Bench")['train']

    resolution = 1024
    pipe = init_edit_mgt(device='cuda:0')
    method_name = args.method_name
    
    for item in tqdm(dataset):
        if item['instruction_language'] != 'en':
            continue
        
        father_path = f"eval/geditbench/results/{method_name}/fullset/{item['task_type']}/en"
        os.makedirs(father_path, exist_ok=True)
        file_save_path = f"{father_path}/{item['key']}.png"
        if os.path.exists(file_save_path):
            continue

        sample = pipe(
            prompt=[item['instruction']],
            negative_prompt=[negative_prompt],
            height=resolution,
            width=resolution,
            num_inference_steps=36,
            guidance_scale=6,
            num_images_per_prompt=1,
            reference_image=[item['input_image'].resize((resolution, resolution))],
            reference_strength=1.1,
        ).images[0]

        w, h = item['input_image'].size
        sample.resize((w, h)).save(file_save_path)