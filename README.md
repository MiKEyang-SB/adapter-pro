# <img src="figure/LOGO2.png" width="40%" style="vertical-align:-7px;" />

⚓ The official implementation of **VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model**

<div id="top" align="center">
<p align="center">
<img src=figure/Framework.png width=95% />
</p>
</div>

> **📝 Paper: https://arxiv.org/abs/2502.19645**<br/>
> **🌍 Project page: https://vla-adapter.github.io/**<br/>
> **🤗 HuggingFace: https://huggingface.co/VLA-Adapter**

<br/>

## 📢 News!
- **[2025/09/21]** We released our codes! An enhanced **Pro** version is also released (this version conforms to the pipeline in the original paper, but is optimized in implementation). Everyone is welcome to use it!🎉
- **[2025/09/13]** Our paper won the 🥇**first place** in the [daily list](https://huggingface.co/papers/date/2025-09-12) and the 🥈**second place** in the [weekly list](https://huggingface.co/papers/week/2025-W37) in HF! 
- **[2025/09/12]** We released the original version of the VLA-Adapter for four LIBERO models on [HuggingFace](https://huggingface.co/VLA-Adapter).
- **[2025/09/11]** We released our paper on [ArXiv](https://arxiv.org/abs/2509.09372).

<br/>

## 📆 TODO List<a name="todo"></a>

- A more powerful version, **VLA-Adapter++**, and a detailed **technical report** will be released soon. This will include better performance, more experimental scenarios, and more real-world systems supported.
- It will soon be compatible with various foundation models, including but not limited to [VPP](https://arxiv.org/abs/2412.14803), [π0.5](https://arxiv.org/abs/2504.16054).


<br/>

## 📝 Table of Contents

<!-- - [:movie_camera: Demo](#movie_camera-demo)
- [:loudspeaker: News](#loudspeaker-news)
- [🤗 Model Zoo](#ckpts)
- [:video_game: Getting Started](#installation)
- [:fire: Training Recipe](#fire-training-recipe)
  - [Data Preparation](#zero-data-preparation)  
  - [Task-centric Latent Action Learning](#one-task-centric-latent-action-learning)
  - [Pretraining of Generalist Policy](#two-pretraining-of-generalist-policy)
  - [Post-training for Deployment & Evaluations](#three-post-training-for-deployment--evaluations)
    - [Real-world Experiment](#mechanical_arm-real-world-experiment)
    - [LIBERO](#1-libero)
    - [CALVIN](#2-calvin)
    - [Room2Room](#3-room2room)
    - [SimplerEnv](#4-simplerenv)
- [:rocket: UniVLA's Performance](#rocket-univlas-performance) -->
- [🚀 Quick Start](#quick-start)  &emsp; => The **conda environment** and **dependencies** of VLA-Adapter are given.
  - [1.1 Conda Environment of VLA-Adapter.](#1.1—conda-environment-of-vla-adapter.)
- [📚 Data Preparation](#data-preparation) &emsp; => Provides the **installation process** and **file structure** of LIBERO and CALVIN environments.
- [Acknowledgment](#acknowledgment)

<br/>

## 🚀 Quick Start


### 1.1 Conda Environment of VLA-Adapter.

```bash
# Create and activate conda environment
conda create -n vla-adapter python=3.10.16 -y
conda activate vla-adapter
```

###  1.2 Install Dependencies.

```bash
# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone vla-adapter repo and pip install to download dependencies
git clone https://github.com/VLA-Adapter/code-for-adatper.git
cd vla-adapter

# Installation may fail on some machines. If it fails, you can solve it by lowering the `setuptools` version: `pip install setuptools==57.5.0`
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
pip install "flash-attn==2.5.5" --no-build-isolation
# If you run into difficulty, try `pip cache remove flash_attn` first, or visit the website to download it. (https://github.com/Dao-AILab/flash-attention/releases/tag/v2.5.5)
# You can download the corresponding `.whl` file according to the cuda version of `nvidia-smi`, and then run `pip install flash_attn-2.5.5+cuXX...whl` to install it. 
# We use the `flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` file.
```

<br/>
<br/>


## 📚 2. Data Preparation <a name="data-preparation"></a>

### 2.1 LIBERO Benchmark.

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO) and required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # From vla-adapter base dir
```

To download the [LIBERO datasets](https://huggingface.co/datasets/openvla/modified_libero_rlds) that we used in our fine-tuning experiments, run the command below. This will download the `Spatial`, `Object`, `Goal`, and `Long` datasets in `RLDS` format, i.e., `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`. (`"_no_noops"` stands for no no-op actions, i.e., training samples with near-zero actions are filtered out). These datasets require `~10GB` of memory in total. If needed, see details on how to download the original non-RLDS datasets [here](https://github.com/openvla/openvla?tab=readme-ov-file#libero-setup). You can use these to fine-tune Prismatic-VLMs (built on Qwen2.5-0.5B) or other VLMs.

```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

### 2.2 CALVIN Benchmark.

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
sh install.sh
```

To download the [CALVIN ABC→D datasets](https://github.com/mees/calvin/tree/main/dataset) that we used in our fine-tuning experiments, run the command below. 

```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```

If you want to download the RLDS format, you can visit [here](https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds) to download it. This dataset require `~50GB` of memory.


### 2.3 Benchmark Location.

The downloaded dataset can be placed in the `/data` folder. The overall directory structure is as follows:

```
·
├── data
·   ├── libero
    │   ├── libero_10_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_goal_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    │   ├── libero_object_no_noops
    │   │   └── 1.0.0  (It contains some json files and 32 tfrecord files)
    │   ├── libero_spatial_no_noops
    │   │   └── 1.0.0  (It contains some json files and 16 tfrecord files)
    │
    ├── calvin_abc
    │   └── 1.0.0  (It contains some json files, 512 train tfrecord files, and 32 valid tfrecord files)
    │
    └── other benchmarks ...

```

<br/>
<br/>

## 🔥 3. Training for Different Configurations

**We provide different training configurations for different users. You can choose the configuration suitable for training based on your GPU card type.**

### 📄 3.1 Related File.
* `vla-scripts/finetune.py`: VLA fine-tuning script


### 📘 3.2 How to Train? <br/> &emsp;&emsp;&emsp;&emsp; => *Extremely Limited VRAM (A card with 10GB-12GB) (e.g. NVIDIA GeForce RTX 2080Ti, 3060, 3080, 4070, 4080, and 5070).*

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If your resources are extremely limited, you can set `--batch_size 1` and `--lora_rank 64`, it only be required `9.6GB` of VRAM. Certainly, `batch size = 1` will cause gradient updates to be greatly affected by extreme values, and loss convergence will be unstable. In this case, you can modify the `grad_accumulation_steps` parameter to simulate a similar effect. For example, `--batch_size 1` with `--grad_accumulation_steps 8` has a similar effect to `--batch_size 8`, but the training speed will be slower. This means that you can't use the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) model on a card with `10GB` because even with `batch size = 1`, it requires `25GB` of VRAM. Fortunately, you can use VLA-Adapter. However, the `batch size` is still small, you can increase `--max_steps` to achieve the performance reported in the paper.

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version`, requiring only `8.6GB` of VRAM. You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 400000 \
--max_steps 400005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 1 \
--grad_accumulation_steps 8 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.

<br/>

### 📘 3.3 How to Train? <br/> &emsp;&emsp;&emsp;&emsp; => *Low VRAM (A card with 24GB) (e.g. NVIDIA GeForce RTX 3090 and 4090).*

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***


If you have such a device, you can increase the `batch size` and `lora rank`: `--batch_size 4` and `--lora_rank 64`. This only takes nearly `20GB`. This is consistent with the rank in our paper. This means that you can't use the [OpenVLA-OFT](https://github.com/moojink/openvla-oft) model on a card with `24GB` because even with `batch size = 1`, it requires `25GB` of VRAM. Fortunately, you can use VLA-Adapter. However, the `batch size` is still small, you can increase `--max_steps` to achieve the performance reported in the paper.

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch), requiring only `17.6GB` of VRAM. You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.


 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 4 \
--grad_accumulation_steps 4 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.



<br/>

### 📘 3.4 How to Train? <br/> &emsp;&emsp;&emsp;&emsp; => *A Consumer GPU with 32GB (e.g. NVIDIA GeForce RTX 5090) <br/> &emsp;&emsp;&emsp;&emsp; => A Professional-Grade GPU with 40GB-48GB (e.g. NVIDIA A100-40GB, A800-40GB, L20, and RTX A6000).*

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

If you have such a device, you can increase the `batch size` and `lora rank`: `--batch_size 8` and `--lora_rank 64`. This only takes nearly `29GB`. 

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.

With this configuration, you can achieve the same results as in our paper on the `LIBERO-Object` benchmark, achieving a `99.2%` success rate, in just `8 hours`. The `LIBERO-Spatial` benchmark requires approximately 10 hours of training. However, the `LIBERO-Long` benchmark takes longer because its tasks are longer and more difficult, requiring more training steps to achieve superior performance.

>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch). You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

 ```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 2 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter--libero_spatial_no_noops--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.



<br/>


### 📘 3.5 How to Train? <br/> &emsp;&emsp;&emsp;&emsp; => *Professional-Grade GPUs with ≥80GB (e.g. NVIDIA A100-80GB, A800-80GB, H100, H800, H20-NVLink, and H200).*

>***About `batch_size`, `lora_rank`, `grad_accumulation_steps`, and `max_steps`.***

You can use 1 to 8 GPUs for training by changing the number of `CUDA_VISIBLE_DEVICES` to the GPU number and the number of GPUs after `--nproc-per-node`. In our paper, we use 4×H100 GPU for training. In this configuration, the four suites of the LIBERO benchmark, `Spatial` (only five hours), `Object` (less than one hour), `Goal` (three hours), and `Long` (half a day); the `CALVIN` benchmark (eight hours)

>***About `vlm_path`.***

The VLM in the VLA-Adapter uses the Prismatic-VLMs architecture, with the LLM backbone being `Qwen2.5-0.5B`. You can download it from https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b and place it in `/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b`.

>***About `data_name`.***

Launch the fine-tuning script with the vla-adapter configuration below. It can run in the background, and the running progress can be seen in the `/logs` folder. You can replace `libero_spatial_no_noops` with `libero_object_no_noops`, `libero_goal_no_noops`, or `libero_10_no_noops`. If you are using the `CALVIN` benchmark, you need to delete `\libero` in `--data_root_dir` and replace `libero_spatial_no_noops` with `calvin_abc`.


>***About `use_pro_version`.***

In addition, we recently released an enhanced version `Pro` of the VLA-Adapter. While its framework remains consistent with the original paper, it has been enhanced in the implementation, resulting in significantly improved performance. **Therefore, we strongly recommend using the Pro version!** The `Pro` version's `Policy` size is `207MB`, and training speed is virtually unchanged. The `original version` is nearly `1GB` smaller than the `pro version` (1 batch). You can choose whether to use the `Pro` version by setting the `use_pro_version` parameter, i.e., the `Pro` version is `--use_pro_version True`.

```bash
data_name=libero_spatial_no_noops

CUDA_VISIBLE_DEVICES=X torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 150000 \
--max_steps 150005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 16 \
--grad_accumulation_steps 1 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--spatial--$current_time \
> logs/VLA-Adapter--spatial--$current_time.log 2>&1 &
```

Please note that the obtained models will be stored in the `/outputs` folder. Each model will take up nearly `3GB` of memory, so you need to reserve enough space. We strongly recommend that you get our trained model from [VLA-Adapter HuggingFace](https://huggingface.co/VLA-Adapter) and place it in this folder for inference.

## 🦾 Evaluations

### 📄 4.1 Related file.
* `experiments/robot/libero/`: LIBERO eval files
  * `run_libero_eval.py`: LIBERO eval script
  * `libero_utils.py`: LIBERO eval utils
* `experiments/robot/`: General eval utils files
  * `openvla_utils.py`: VLA-specific eval utils
  * `robot_utils.py`: Other eval utils

<br/>

### 🤗 4.2 Checkpoint of VLA-Adapter.
We fine-tuned `Qwen2.5-0.5B` with our adapter bridge paradigm on four LIBERO task suites independently: `LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, and `LIBERO-Long`. 
The four VLA-Adapter checkpoints for LIBERO are available on Hugging Face:
* [VLA-Adapter/LIBERO-Spatial](https://huggingface.co/VLA-Adapter/LIBERO-Spatial) 
* [VLA-Adapter/LIBERO-Object](https://huggingface.co/VLA-Adapter/LIBERO-Object)
* [VLA-Adapter/LIBERO-Goal](https://huggingface.co/VLA-Adapter/LIBERO-Goal)
* [VLA-Adapter/LIBERO-Long](https://huggingface.co/VLA-Adapter/LIBERO-Long)

In addition, we also provide a `Pro` version, we used `4*H100` GPUs for training, `--batch_size 16`, `--lora rank 64`, and the `--max_steps 100000`. The Pro checkpoints is:

* [VLA-Adapter/LIBERO-Spatial-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Spatial-Pro) `(97.8 -> 99.4)`
* [VLA-Adapter/LIBERO-Object-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Object) `(99.2 -> 99.4)`
* [VLA-Adapter/LIBERO-Goal-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Goal) `(97.2 -> 97.8)`
* [VLA-Adapter/LIBERO-Long-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Long-Pro) `(95.0 -> 96.4)`
* [VLA-Adapter/CALVIN-ABC-Pro](https://huggingface.co/VLA-Adapter/LIBERO-Long-Pro) `(4.42 -> 4.50)`

These files need to be placed in the `/output` folder. If you trained your own models, it will also be stored here. The subsequent eval code will call the model in this folder for inference.


<br/>


### 📘 4.3 How to Eval?
**We strongly recommend that you use our open source Pro version of the model, which has stronger performance.** To start evaluations with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above. If you want to use the original version of the model, you only need to adjust the `-- use_pro_version` parameter to `False` and pass the original version of the model to the `--pretrained_checkpoint` parameter. Finally, the inference results will be displayed in the `/eval_logs` folder, and the inference video will be displayed in the `/rollouts/vla-adapter` folder. 


```bash
# Launch LIBERO-Spatial evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  > eval_logs/Spatial--chkpt.log 2>&1 &


# Launch LIBERO-Object evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Object-Pro \
  --task_suite_name libero_object \
  --use_pro_version True \
  > eval_logs/Object--chkpt.log 2>&1 &


# Launch LIBERO-Goal evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Goal-Pro \
  --task_suite_name libero_goal \
  --use_pro_version True \
  > eval_logs/Goal--chkpt.log 2>&1 &


# Launch LIBERO-Long (LIBERO-10) evals
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-long-Pro \
  --task_suite_name libero_10 \
  --use_pro_version True \
  > eval_logs/Long--chkpt.log 2>&1 &


# Launch CALVIN ABC→D evals
CUDA_VISIBLE_DEVICES=0 python vla-scripts/evaluate_calvin.py \
  --pretrained_checkpoint outputs/CALVIN \
  --use_diffusion False \
  --use_x0_prediction False \
  --use_pro_version True \
  > eval_logs/CALVIN--ABC.log 2>&1 &
```

The evaluation script will run 500 trials by default (10 tasks x 50 episodes each) in LIBERO and 1000 task sequences in CALVIN. Note that results may vary slightly if you use a different GPU than the H100. 

If you want to get the inference **throughput**, you can run it in the `run_libero_eval.py` file. You can add  `start = time.time()` and `end = time.time()` before and after `lines 334--345` and calculate the difference between the two. This difference is the time it takes to generate `8 chunks`. This gives you the inference throughput. We measured it multiple times and took the average value of `0.036s`.


## Success Rate Comparison

<table>
  <tr>
  <td><strong>Category</strong>
   </td>
   <td><strong>Methods</strong>
   </td>
   <td><strong>Scale</strong>
   </td>
   <td><strong>Spatial</strong>
   </td>
   <td><strong>Object</strong>
   </td>
   <td><strong>Goal</strong>
   </td>
   <td><strong>Long</strong>
   </td>
  <td><strong>Avg.</strong>
   </td>
  </tr>
  <tr>
    <td rowspan="10">Large-scale</td>
   <td>FlowVLA (Zhong et al., 2025)</td>
   <td>8.5B</td><td>93.2</td><td>95.0</td><td>91.6</td><td>72.6</td><td>88.1</td>
  </tr>

   <tr>
   <td>UnifiedVLA (Wang et al., 2025)</td>
   <td>8.5B</td><td>95.4</td><td> <i><u>98.8*</u></i></td><td> 93.6 </td><td>94.0 </td><td>95.5</td>
  </tr>

  <tr>
   <td>OpenVLA (Kim et al., 2024)</td>
   <td>7B</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>76.5</td>
  </tr>

  <tr>
   <td>OpenVLA-OFT (Kim et al., 2025)</td>
   <td>7B</td><td><i><u>97.6*</u></i></td><td>98.4</td><td><b>97.9</b></td><td><i><u>94.5*</u></i></td><td><i><u>97.1*</u></i></td>
  </tr>

  <tr>
   <td>UniVLA (Bu et al., 2025)</td>
   <td>7B</td><td>96.5</td><td> 96.8</td><td> 95.6 </td><td>92.0 </td><td>95.2</td>
  </tr>

  <tr>
   <td>CoT-VLA (Zhao et al., 2025)</td>
   <td>7B</td><td>87.5 </td><td>91.6 </td><td>87.6</td><td> 69.0</td><td> 81.1</td>
  </tr>

  <tr>
   <td>WorldVLA (Cen et al., 2025)</td>
   <td>7B</td><td>87.6</td><td> 96.2</td><td> 83.4</td><td> 60.0</td><td> 81.8</td>
  </tr>

  <tr>
   <td>TraceVLA (Zheng et al., 2025)</td>
   <td>7B</td><td>84.6</td><td> 85.2</td><td> 75.1</td><td> 54.1</td><td> 74.8</td>
  </tr>

  <tr>
   <td>MolmoAct (Lee et al., 2025)</td>
   <td>7B</td><td>87.0</td><td> 95.4 </td><td>87.6</td><td> 77.2 </td><td>86.6</td>
  </tr>

  <tr>
   <td>ThinkAct (Huang et al., 2025)</td>
   <td>7B</td><td>88.3 </td><td>91.4</td><td> 87.1</td><td> 70.9</td><td> 84.4</td>
  </tr>

  <tr>
  <td rowspan="7">Small-scale</td>
   <td>4D-VLA (Zhang et al., 2025)</td>
   <td>4B</td><td>88.9</td><td> 95.2</td><td> 90.9</td><td> 79.1 </td><td>88.6</td>
  </tr>

  <tr>
   <td>SpatialVLA (Qu et al., 2025)</td>
   <td>4B</td><td>88.2</td><td> 89.9</td><td> 78.6</td><td> 55.5 </td><td>78.1</td>
  </tr>

  <tr>
   <td>π0 (Black et al., 2024)</td>
   <td>3B</td><td>96.8</td><td> <i><u>98.8*</u></i> </td><td>95.8</td><td> 85.2</td><td> 94.2</td>
  </tr>

  <tr>
   <td>π0-FAST (Pertsch et al., 2025)</td>
   <td>3B</td><td>96.4</td><td> 96.8 </td><td>88.6</td><td> 60.2</td><td> 85.5</td>
  </tr>

  <tr>
   <td>NORA (Hung et al., 2025)</td>
   <td>3B</td><td>92.2 </td><td>95.4 </td><td>89.4</td><td> 74.6 </td><td>87.9</td>
  </tr>

  <tr>
   <td>SmolVLA (Shukor et al., 2025)</td>
   <td>2.2B</td><td>93.0</td><td> 94.0 </td><td>91.0</td><td> 77.0 </td><td>88.8</td>
  </tr>

  <tr>
   <td>GR00T N1 (NVIDIA et al., 2025)</td>
   <td>2B</td><td>94.4</td><td> 97.6 </td><td>93.0 </td><td>90.6</td><td> 93.9</td>
  </tr>

  <tr>
    <td rowspan="5">Tiny-scale</td>
   <td>Seer (Tian et al., 2025)</td>
   <td>0.57B</td><td>-</td><td> - </td><td>- </td><td>78.7</td><td> 78.7</td>
  </tr>

  <tr>
   <td>VLA-OS (Gao et al., 2025)</td>
   <td>0.5B</td><td>87.0 </td><td>96.5</td><td> 92.7 </td><td>66.0</td><td> 85.6</td>
  </tr>

  <tr>
   <td>Diffusion Policy (Chi et al., 2023)</td>
   <td>-</td><td>78.3</td><td> 92.5</td><td> 68.3 </td><td>50.5 </td><td>72.4</td>
  </tr>

  <tr>
   <td><b>VLA-Adapter (Ours)</b></td>
   <td><b>0.5B</b></td><td><b>97.8</b></td><td> <b>99.2</b> </td><td><i><u>97.2*</u></i></td><td> <b>95.0 </b></td><td><b>97.3</b></td>
  </tr>

  <tr>
   <td><b>VLA-Adapter (Pro)</b></td>
   <td><b>0.5B</b></td><td><b>99.4</b></td><td> <b>99.4</b> </td><td><i><u>97.8*</u></i></td><td> <b>96.4</b></td><td><b>98.3</b></td>
  </tr>
  
</table>


## Effectiveness Comparison

<img src=figure/Teaser1.png width=70% />




## Citation<a name="cite"></a>

### 🫶 If you feel that this paper, models, or codes are helpful, please cite our paper, thanks for your support of VLA-Adapter!

```bibtex
@article{wang2025vlaadapter,
  author={Wang, Yihao and Ding, Pengxiang and Li, Lingxiao and Cui, Can and Ge, Zirui and Tong, Xinyang and Song, Wenxuan and Zhao, Han and Zhao, Wei and Hou, Pengxu and Huang, Siteng and Tang, Yifan and Wang, Wenhui and Zhang, Ru and Liu, Jianyi and Wang, Donglin},
  title={VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model},
  journal={arXiv preprint arXiv:2509.09372},
  year={2025}
}
```

## 📚 Acknowledgment <a name="acknowledgment"></a>

We thank [OpenVLA-OFT](https://github.com/moojink/openvla-oft) for their open-sourced work!