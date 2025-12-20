<h1 align="center" style="line-height: 50px;">
  ✨ EditMGT: Unleashing the Potential of Masked Generative Transformer in Image Editing ✨
</h1>

<div align="center">
Wei Chow<sup>1,*</sup>, Linfeng Li<sup>1,*</sup>, Lingdong Kong<sup>1</sup>, Zefeng Li<sup>1</sup>, Qi Xu<sup>1</sup>, Hang Song<sup>1</sup>, Tian Ye<sup>4</sup>, Xian Wang<sup>1</sup>, Jinbin Bai<sup>3</sup>, Shilin Xu<sup>1</sup>, Xiangtai Li<sup>1</sup>, Junting Pan<sup>1</sup>, Shaoteng Liu<sup>1</sup>, Ran Zhou<sup>1</sup>, Tianshu Yang<sup>1</sup>, Songhua Liu<sup>2</sup>

<sup>1</sup>ByteDance, <sup>2</sup>Shanghai Jiao Tong University, <sup>3</sup>National University of Singapore, <sup>4</sup>The Hong Kong University of Science and Technology (Guangzhou)
  
<small>*Equal Contribution</small>

[![arXiv](https://img.shields.io/badge/arXiv-2512.11715-b31b1b.svg)](https://arxiv.org/abs/2512.11715)
[![Dataset](https://img.shields.io/badge/🤗%20CrispEdit2M-Dataset-yellow)](https://huggingface.co/datasets/WeiChow/CrispEdit-2M)
[![Checkpoint](https://img.shields.io/badge/🧨%20EditMGT-CKPT-blue)](https://huggingface.co/WeiChow/EditMGT)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/weichow23/editmgt/tree/main)
[![Page](https://img.shields.io/badge/🏠%20Home-Page-b3.svg)](https://weichow23.github.io/editmgt/)
</div>

## 🚀 Project Introduction

EditMGT is a novel framework that leverages Masked Generative Transformers for advanced image editing tasks. Our approach enables precise and controllable image modifications while preserving original content integrity.

## ⚡ Quick Start  

First, clone the repository and navigate to the project root:  
```shell
git clone https://github.com/weichow23/editmgt
cd editmgt
```

## 🔧 Environment Setup

```bash
# Create and activate conda environment
conda create --name editmgt python=3.9.2
conda activate editmgt

# Optional: Install system dependencies
sudo apt-get install libgl1-mesa-glx libglib2.0-0 -y

# Install Python dependencies
pip3 install git+https://github.com/openai/CLIP
pip3 install -r requirements.txt
```
⚠️ **Note**: If you encounter any strange environment library errors, please refer to [Issues](https://github.com/viiika/Meissonic/issues/14) to find the correct version that might fix the error.

## 🎯 Training

For training, you can try using `--instance_dataset HuggingFaceDataset` with the [AnyEdit](https://huggingface.co/datasets/Bin1117/AnyEdit) dataset to get the training code running. After verifying functionality, you can switch to your own dataset.

### 📝 Configuration
The `train/train_edit.yaml` file controls training behavior. Key settings include:

| Parameter | Description | Default |
|-----------------|-----------------------------------------------------------------------------|----------|
| `mixed_precision` | Training Precision | `no` for float32; `bf16` for bfloat16 |
| `resume_from_checkpoint` | Resume Training | `direct local path` or `latest` |
| `wandb_id` | Weights & Biases ID | keep empty or id number |
| `pretrained_model_name_or_path` | Base Model | `MeissonFlow/Meissonic` |
| `output_dir` | Checkpoint Save Location | `./runs/editmgt` |
| `train_batch_size` | Batch Size per GPU | `4` |
| `gradient_accumulation_steps` | Steps before Weight Update | `8` |
| `learning_rate` | Initial Learning Rate | `1e-4` |
| `max_grad_norm` | Gradient Clipping Value | `10` |
| `text_encoder_architecture` | Text Encoder Model | `CLIP_Gemma2` |
| `resolution` | Image Resolution | `1024` |
| `lr_scheduler` | Learning Rate Schedule | `constant` |
| `max_train_steps` | Total Training Steps | `500000` |
| `checkpointing_steps` | Save Frequency | `200` |
| `logging_steps` | Log Metrics Frequency | `10` |

To use Weights & Biases for tracking the training process, add these to your environment:
```shell
echo 'export WANDB_API_KEY=<YOUR WANDB API>' >> ~/.bashrc
echo 'export WANDB_ENTITY=<YOUR WANDB ENTITY>' >> ~/.bashrc
echo 'export hf_token=<YOUR HF TOKEN>' >> ~/.bashrc
source ~/.bashrc
```

#### 🧩 Enabling LoRA
To use LoRA (reduces memory usage by ~70%), add the following to the `train.sh` file under `train`:
```shell
--use_lora \
--lora_r 32 \
--lora_alpha 128
```
Here, `lora_r` and `lora_alpha` represent the rank and alpha parameters of LoRA, respectively.

#### 🚗 Run Training
Use the script to start training. Specify the config file and enable LoRA if needed:
```shell
bash train/train.sh
```
⚠️ **Important**: Since Meissonic has not released bf16 weights, fp16 weights may cause training instability. Therefore, we use fp32 for training (although this will be slower).

## 🔍 Inference
We provide a standard example script:

```shell
python3 infer.py
```

### 🃏 GEditBench-EN
We also provide evaluation scripts for GEditBench. Refer to the [Official Repo](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md) for implementation details.

You'll need to install additional dependencies beyond our `requirements.txt`:
```shell
pip3 install megfile==4.1.4
```
Then use the scirpt:

```shell
python3 eval/geditbench/infer.py
```

Generate and organize your images in the following directory structure:
```
results/
├── {method_name}/
│   └── fullset/
│       └── {edit_task}/
│           └── en/  # English instructions
│               ├── key1.png
│               ├── key2.png
│               └── ...
```

Run the inference script:
```shell
PYTHON_PATH='./' python3 eval/geditbench/infer.py
```

For GPT-4.1 evaluation, set up your API keys in `eval/geditbench/rate.py` at lines `L103` and `L105` for GPT4.1 access, then run:

```shell
PYTHON_PATH='./' python3 eval/geditbench/rate.py --model_name editmgt --save_dir eval/geditbench/score_dir --backbone gpt4.1 --edited_images_dir eval/geditbench/results/ --instruction_language en
```

Run the analysis script to get scores for semantics, quality, and overall performance:
```shell
PYTHON_PATH='./' python3 eval/geditbench/stat.py --model_name editmgt --save_path eval/geditbench/score_dir --backbone gpt4o --language en
```

This will output scores broken down by edit category and provide aggregate metrics.

**Note**: GEditBench scores can fluctuate significantly due to random generation results and GPT version differences. Fluctuations around our reported scores are normal. For AnyBench, EmuEdit, and MagicBrush, we recommend reducing guidance scale and steps for the first two, and using our mask strategy for the latter.

## 📄 License
This project is licensed under the `CCBY-4.0` License. See `LICENSE` for details.

## 📑 Citation
```
@article{chow2025editmgt,
  title={EditMGT: Unleashing Potentials of Masked Generative Transformers in Image Editing},
  author={Chow, Wei and Li, Linfeng and Kong, Lingdong and Li, Zefeng and Xu, Qi and Song, Hang and Ye, Tian and Wang, Xian and Bai, Jinbin and Xu, Shilin and others},
  journal={arXiv preprint arXiv:2512.11715},
  year={2025}
}
```

## 🙏 Acknowledgements
We thank all contributors and the research community for their valuable feedback and support.
