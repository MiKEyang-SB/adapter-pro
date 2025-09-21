"""Code to evaluate Calvin."""
import argparse
import json
import logging
import os
# os.environ['PYTHONPATH'] = '/root/RoboDual:' + os.environ.get('PYTHONPATH', '')
from collections import deque
# from peft import PeftModel
from pathlib import Path
# import sys
# sys.path.insert(0, '/root/RoboDual')
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
# from openvla.prismatic.vla.action_tokenizer import ActionTokenizer
# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
# sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from vla_evaluation import DualSystemCalvinEvaluation

from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
# from ema_pytorch import EMA
from transformers.modeling_outputs import CausalLMOutputWithPast
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
# logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
os.environ["CALVIN_ROOT"] = "calvin"
CALVIN_ROOT = os.environ['CALVIN_ROOT']

from collections import Counter
import json
import numpy as np
from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass
import draccus

import os
import torch



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/ssdwork/Pengxiang/code/openvla-oft/outputs/calvin/openvla-7b+calvin_abc_rlds+b20+lr-0.0001+lora-r64+dropout-0.0--image_aug--oft_calvin_parallel_dec--8_acts_chunk--continuous_acts--L1_regression--rank64--lr1e-4--lora--3rd_person_img--2025-05-26_05-39-02--40000_chkpt"     # Pretrained checkpoint path

    use_minivla: bool = False                   # If True, 

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    use_x0_prediction: bool = False
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    # env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)
#################################################################################################################
    # CALVIN
    #################################################################################################################
    calvin_path: str = "calvin"
    log_dir: str = "log"
    with_depth: bool = True
    with_gripper: bool = True
    with_cfg: bool = True
    enrich_lang: bool = False
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                 # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "Pro"                        # version of exps

def print_and_save(results, sequences, eval_result_path, task_name=None, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    # model_name = 'vla-test'
    if not os.path.isdir(f'./{task_name}'):
        os.mkdir(f'./{task_name}')
    with open(f'./{task_name}/split_{torch.cuda.current_device()}.json', "w") as file:
        json.dump(chain_sr, file)

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(eval_result_path, "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def make_env(dataset_path, observation_space, device):
    val_folder = Path(dataset_path) / "validation"
    from calvin_env_wrapper import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, num_procs, procs_id, eval_dir, ep_len, num_sequences, task_name='test', enrich_lang=False, debug=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)


    if enrich_lang:
        with open('/root/RoboDual/vla-scripts/enrich_lang_annotations.json', 'r') as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    # val_annotations = {key: val for key, val in val_annotations.items() if 'push' in key and 'right' in key}  # 只保留了push something right的任务！！！！！！！！
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)

    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs * procs_id:num_seq_per_procs * (procs_id + 1)]

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line = f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, task_name, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout_hi3(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:  # return 5!!!!!!!!!!!!!!!!!!!
            # print('success: ', subtask_i)
            success_counter += 1
        else:
            return success_counter
    return success_counter


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        # normalized_action[..., -1] = np.sign(normalized_action[..., -1])
        sign = np.sign(normalized_action[..., -1])
        sign = np.array(sign)  # Ensure it is an array and not a scalar
        sign[sign == 0.0] = 1  # Change 0 to 1
        sign[sign == -0.0] = -1  # Change -0 to -1
        normalized_action[..., -1] = sign

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    img_dict = {
        'static': [],
        'gripper': [],
    }
    action_queue = deque(maxlen=8)

    for step in range(ep_len):
        if len(action_queue) == 0:
            actions = model.step(obs, lang_annotation, step)
            action_queue.extend(actions)

        action = action_queue.popleft()  # {ndarray: (7,)}
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())


        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

    print(colored("fail", "red"), end=" ")
    for key in img_dict.keys():
        clip = ImageSequenceClip(img_dict[key], fps=50)
        clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-fail.mp4'), fps=50, codec='libx264', bitrate="5000k")
    return False

import os
import time
import copy
import numpy as np
from moviepy.editor import ImageSequenceClip
from termcolor import colored

def rollout_act_style(
    env,
    model,
    task_oracle,
    subtask,
    val_annotations,
    debug,
    eval_dir,
    subtask_i,
    sequence_i,
    ep_len,
    chunk_size=8,           # H：每次模型预测的动作块长度
    stride=1,               # K：每次执行的步数（建议 1）
    window_type='cosine',   # 融合窗：'linear' 或 'cosine'
    clamp_action=None,      # 例如 (-1, 1)；None 则不裁剪
    fps=50,                 # 导出视频帧率
    codec='libx264',        # 视频编码器
    bitrate='5000k',        # 比特率
    write_audio=False       # 设为 False 避免音频处理问题
):
    """
    ACT 风格的 rollout：
    - 每个时间步根据当前观测预测一个动作块 new_chunk（长度 H=chunk_size）
    - 使用时间相关的加权窗将 prev_chunk 与 new_chunk 在重叠区融合，得到 fused_chunk
    - 执行 fused_chunk 的前 K（stride）步动作
    - 将 fused_chunk 左移 K 位作为下一步的 prev_chunk
    - 每一步执行后检查任务成功；成功或 done 则提前结束
    """

    assert chunk_size >= 2, "chunk_size 应 >= 2"
    assert 1 <= stride <= chunk_size, "stride 应在 [1, chunk_size] 区间"
    if window_type not in ('linear', 'cosine'):
        raise ValueError("window_type 仅支持 'linear' 或 'cosine'")

    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.2)

    # 初始化环境与模型
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    # 视频帧缓存（整个 episode 级别）
    img_dict = {
        'static': [],
        'gripper': [],
    }

    # 构造融合窗权重（长度为 H，越靠近现在 prev 更重，越靠未来 new 更重）
    H = chunk_size
    if window_type == 'linear':
        w_new = np.linspace(0.0, 1.0, H, dtype=np.float32)  # 0 -> 1
    else:  # cosine
        x = np.linspace(0.0, np.pi, H, dtype=np.float32)
        w_new = (1.0 - np.cos(x)) / 2.0
    w_prev = 1.0 - w_new
    # 为广播到 [H, act_dim] 做准备
    w_new = w_new[:, None]
    w_prev = w_prev[:, None]

    prev_chunk = None  # 上一轮融合后的“未来动作块”，形状 [H, act_dim]
    success = False

    # 每个外层循环会执行最多 stride 步
    t = 0
    while t < ep_len:
        # 1) 基于当前观测预测新块
        new_chunk = model.step(obs, lang_annotation, t)
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        if new_chunk.ndim == 1:
            # 容错：若模型误返回单步动作，扩展为 1xA，并在下方 pad 成 HxA
            new_chunk = new_chunk[None, :]
        assert new_chunk.shape[0] <= H, "model.step 返回序列长度不能超过 chunk_size"
        act_dim = new_chunk.shape[1]

        # 若模型输出不足 H，尾部用最后一个动作重复填满 H
        if new_chunk.shape[0] < H:
            tail = np.repeat(new_chunk[-1:,:], H - new_chunk.shape[0], axis=0)
            new_chunk = np.concatenate([new_chunk, tail], axis=0)

        # 2) 与 prev_chunk 融合
        if prev_chunk is None or prev_chunk.shape != new_chunk.shape:
            fused_chunk = new_chunk
        else:
            # 时间位置加权融合
            fused_chunk = w_prev * prev_chunk + w_new * new_chunk

        # 3) 执行 fused_chunk 的前 K（stride）步
        steps_this_iter = min(stride, ep_len - t)
        for k in range(steps_this_iter):
            current_action = fused_chunk[k]

            # 可选：裁剪到动作空间
            if clamp_action is not None:
                low, high = clamp_action
                current_action = np.clip(current_action, low, high)

            # 处理并执行
            processed_action = process_action(current_action, "openvla")
            obs, reward, done, current_info = env.step(processed_action.tolist())

            # 记录图像帧
            # 使用 np.copy 比 deepcopy 更高效，假设 obs['rgb_obs'][...] 是 numpy 数组
            img_dict['static'].append(np.copy(obs['rgb_obs']['rgb_static']))
            img_dict['gripper'].append(np.copy(obs['rgb_obs']['rgb_gripper']))

            # 成功检测
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                success = True
                break

            # 如果环境 done 也提前结束
            if done:
                break

        # 步计数推进
        t += steps_this_iter

        # 若成功或 done，跳出主循环
        if success or done:
            break

        # 4) 更新 prev_chunk：将 fused_chunk 左移 K 位，末尾用最后一个值填充，保持长度 H
        if steps_this_iter < H:
            # 左移
            tail_fill = np.repeat(fused_chunk[-1:, :], steps_this_iter, axis=0)
            shifted = np.concatenate([fused_chunk[steps_this_iter:], tail_fill], axis=0)
            prev_chunk = shifted
        else:
            # steps_this_iter == H：整块已执行完，下一轮从新块开始
            prev_chunk = None

    # 写出结果视频
    try:
        status = "succ" if success else "fail"
        color = "green" if success else "red"
        print(colored(status, color), end=" ")

        # 导出两路相机视频
        for key in img_dict.keys():
            if len(img_dict[key]) > 0:
                clip = ImageSequenceClip(img_dict[key], fps=fps)
                out_path = os.path.join(
                    eval_dir,
                    f'{sequence_i}-{subtask_i}-{subtask}-{key}-{status}.mp4'
                )
                clip.write_videofile(out_path, fps=fps, codec=codec, bitrate=bitrate, audio=write_audio, logger=None)
    except Exception as e:
        # 不影响返回值，但打印告警
        print(colored(f"[video export error] {e}", "yellow"))

    return success


def rollout_hi3(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    img_dict = {
        'static': [],
        'gripper': [],
    }

    for step in range(80):
        # 初始化3×8的动作缓冲区
        action_buffers = [None, None, None]

        # 第0步，生成q0
        action_buffers[0] = model.step(obs, lang_annotation, 0)  # 8个动作
        action = action_buffers[0][0]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        # 检查任务完成
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        # 第1步，生成q1
        action_buffers[1] = model.step(obs, lang_annotation, 1)
        action = (action_buffers[0][1] + action_buffers[1][0]) / 2
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        # 第2步，生成q2
        action_buffers[2] = model.step(obs, lang_annotation, 2)
        action = (action_buffers[0][2] + action_buffers[1][1] + action_buffers[2][0]) / 3
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        # 第3~8步：常规三队列平均
        for t in range(2, 7):
            action = (action_buffers[0][t] + action_buffers[1][t-1] + action_buffers[2][t-2]) / 3
            action = process_action(action, "openvla")
            obs, reward, done, current_info = env.step(action.tolist())

            img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
            img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                print(colored("success", "green"), end=" ")
                for key in img_dict.keys():
                    clip = ImageSequenceClip(img_dict[key], fps=50)
                    clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
                return True

        # 第9步：两项平均
        action = (action_buffers[1][7] + action_buffers[2][6]) / 2
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

        # 第10步：直接用最后一个动作
        action = action_buffers[2][7]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(action.tolist())

        img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print(colored("success", "green"), end=" ")
            for key in img_dict.keys():
                clip = ImageSequenceClip(img_dict[key], fps=50)
                clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'), fps=50, codec='libx264', bitrate="5000k")
            return True

    # 全部失败
    print(colored("fail", "red"), end=" ")
    for key in img_dict.keys():
        clip = ImageSequenceClip(img_dict[key], fps=50)
        clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-fail.mp4'), fps=50, codec='libx264', bitrate="5000k")
    return False

def rollout_hi4(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)

    os.makedirs(eval_dir, exist_ok=True)

    def tolist(a):
        return np.asarray(a).tolist()

    def save_video(tag, img_dict):
        for key in img_dict.keys():
            clip = ImageSequenceClip(img_dict[key], fps=50)
            out = os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-{tag}.mp4')
            clip.write_videofile(out, fps=50, codec='libx264', bitrate="5000k", verbose=False, logger=None)
            clip.close()

    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    img_dict = {
        'static': [],
        'gripper': [],
    }

    def record_obs(o):
        img_dict['static'].append(copy.deepcopy(o['rgb_obs']['rgb_static']))
        img_dict['gripper'].append(copy.deepcopy(o['rgb_obs']['rgb_gripper']))

    def succeeded(current_info):
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        return len(current_task_info) > 0

    record_obs(obs)

    # 每个 block 里尝试“4队列×8动作”的组合，总共执行 13 步（0..12）
    # 外层最多尝试 50 个 block
    for block in range(50):
        action_buffers = [None, None, None, None]  # 4 个队列

        # 步0：生成 q0，执行 q0[0]
        action_buffers[0] = model.step(obs, lang_annotation, 0)
        action = action_buffers[0][0]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 步1：生成 q1，执行 (q0[1] + q1[0]) / 2
        action_buffers[1] = model.step(obs, lang_annotation, 1)
        action = (action_buffers[0][1] + action_buffers[1][0]) / 2
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 步2：生成 q2，执行 (q0[2] + q1[1] + q2[0]) / 3
        action_buffers[2] = model.step(obs, lang_annotation, 2)
        action = (action_buffers[0][2] + action_buffers[1][1] + action_buffers[2][0]) / 3
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 步3：生成 q3，执行 (q0[3] + q1[2] + q2[1] + q3[0]) / 4
        action_buffers[3] = model.step(obs, lang_annotation, 3)
        action = (action_buffers[0][3] + action_buffers[1][2] + action_buffers[2][1] + action_buffers[3][0]) / 4
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 常规“四队列滑动平均”：t = 3..6（共4步）
        # 执行 (q0[t] + q1[t-1] + q2[t-2] + q3[t-3]) / 4
        for t in range(3, 7):
            action = (action_buffers[0][t] + action_buffers[1][t-1] + action_buffers[2][t-2] + action_buffers[3][t-3]) / 4
            action = process_action(action, "openvla")
            obs, reward, done, current_info = env.step(tolist(action))
            record_obs(obs)
            if succeeded(current_info):
                print(colored("success", "green"), end=" ")
                save_video('succ', img_dict)
                return True

        # 后尾部收束（与三队列版本的“第9步、第10步”类似），逐步减少参与平均的队列数量
        # 步8：使用 q1[7], q2[6], q3[5] 三项平均
        action = (action_buffers[1][7] + action_buffers[2][6] + action_buffers[3][5]) / 3
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 步9：使用 q2[7], q3[6] 两项平均
        action = (action_buffers[2][7] + action_buffers[3][6]) / 2
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 步10：使用 q3[7] 直接执行
        action = action_buffers[3][7]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 可选：再多两步“外推”以充分利用已执行后的观测（若希望固定为 13 步，也可以在上一步结束）
        # 这里保持与三队列版的总步数扩展为 13 步（0..12），用保守策略重复最后动作两次或做平滑过渡
        # 步11：重复 q3[7]（或进行一次小幅平滑）
        action = action_buffers[3][7]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

        # 步12：再次重复 q3[7]
        action = action_buffers[3][7]
        action = process_action(action, "openvla")
        obs, reward, done, current_info = env.step(tolist(action))
        record_obs(obs)
        if succeeded(current_info):
            print(colored("success", "green"), end=" ")
            save_video('succ', img_dict)
            return True

    # 全部失败
    print(colored("fail", "red"), end=" ")
    save_video('fail', img_dict)
    return False

# 辅助函数
def update_image_data(img_dict, obs):
    img_dict['static'].append(copy.deepcopy(obs['rgb_obs']['rgb_static']))
    img_dict['gripper'].append(copy.deepcopy(obs['rgb_obs']['rgb_gripper']))

def check_success(start_info, current_info, subtask, task_oracle):
    return len(task_oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0

def handle_success(img_dict, eval_dir, sequence_i, subtask_i, subtask):
    print(colored("success", "green"), end=" ")
    for key in img_dict.keys():
        clip = ImageSequenceClip(img_dict[key], fps=50)
        clip.write_videofile(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-{key}-succ.mp4'),
                            fps=50, codec='libx264', bitrate="5000k")
    return True





from huggingface_hub import HfApi, hf_hub_download
import shutil
from datetime import datetime
import filecmp
from typing import Any, Dict, List, Optional, Tuple, Union
def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)




def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key

# ====================================================================================================
# 目的：加载与VLA模型相关的 proprio_projector，proprio指有关机器人的位置信息、动作、姿态等信息
# 输入：cfg (pretrained_checkpoint)、llm_dim、proprio_dim
# 输出：初始化并加载了预训练权重的 ProprioProjector 对象




DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
import wandb
def setup_logging(cfg):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id



def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )



MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    # if "image_aug" in str(cfg.pretrained_checkpoint):
    #     assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"


def get_image_resize_size(model_family) -> Union[int, tuple]:
    return MODEL_IMAGE_SIZES[model_family]

def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        # check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor

@draccus.wrap()
def main(cfg: GenerateConfig):
    seed_everything(cfg.seed)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    # device = acc.device
    validate_config(cfg)
    
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)


    # # Load generalist policy
    # processor = AutoProcessor.from_pretrained(args.VLM_backbone, trust_remote_code=False)
    # model = AutoModelForVision2Seq.from_pretrained(
    #     args.VLM_backbone,
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=None,
    #     trust_remote_code=False,
    #     low_cpu_mem_usage=True,
    # )

    # model.vision_backbone.set_num_images_in_input(2)  # OFT-add!
    # model.eval()
    # model = model.to(device)
    # _load_dataset_stats(model, args.VLM_backbone)

    # # Load proprio projector and action head if needed
    # proprio_projector = get_proprio_projector(        # OFT-add!
    #     args.VLM_backbone,
    #     model.llm_dim,
    #     proprio_dim=8
    # )
    # action_head = get_action_head(                    # OFT-add!
    #     args.VLM_backbone,
    #     model.llm_dim
    # )

    current_time=time.strftime("%Y-%m-%d_%H-%M-%S")

    save_path = f'./evaluation_results'
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],  # rgb_tactile
        'depth_obs': ['depth_static', 'depth_gripper'],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']}
    eval_dir = save_path + f'/calvin/{current_time}_{cfg.pretrained_checkpoint.split("/")[-1]}/'
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env(os.path.join(CALVIN_ROOT, 'dataset/task_ABC_D'), observation_space, DEVICE)

    # eva = DualSystemCalvinEvaluation(dual_sys, processor, action_tokenizer)
    # dual_sys.eval()

    eva = DualSystemCalvinEvaluation(model, proprio_projector, noisy_action_projector, action_head, processor, use_x0_prediction=cfg.use_x0_prediction)
    avg_reward = torch.tensor(evaluate_policy(
        eva,
        env,
        eval_dir + 'success_rate.txt',
        eval_dir + 'result.txt',
        acc.num_processes,
        acc.process_index,
        eval_dir=eval_dir,
        ep_len=360,
        num_sequences=1000,
        enrich_lang=cfg.enrich_lang,
        debug=False,
    )).float().mean().to(DEVICE)

    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print('average success rate ', avg_reward)


if __name__ == "__main__":
    main()
