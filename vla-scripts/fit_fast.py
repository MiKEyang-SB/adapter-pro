"""
Training script for FAST action tokenizer on LIBERO dataset.

This script loads LIBERO HDF5 data, extracts action sequences,
splits them into subsegments using sliding window, and trains the FAST tokenizer.

Usage:
    python vla-scripts/fit_fast.py \
        --libero_data_dir /path/to/libero/datasets \
        --task_suite libero_spatial_no_noops \
        --T 8 \
        --scale 50 \
        --save_path /path/to/save/tokenizer
"""

import argparse
import os
import h5py
import numpy as np
from transformers import AutoProcessor
from tqdm import tqdm


def split_action_into_subsegments(action, T):
    """
    Split action matrix into consecutive subsegments of length T.
    Uses sliding window with step size 1.

    Args:
        action: Action array of shape (N, action_dim), e.g., (N, 7)
        T: Length of each subsegment (chunk size)

    Returns:
        subsegments: Array of shape (N-T+1, T, action_dim)
    """
    N = len(action)
    if N < T:
        return np.array([])

    subsegments = []

    # Sliding window with step size 1
    for start in range(N - T + 1):
        subsegment = action[start:start+T]
        subsegments.append(subsegment)

    return np.array(subsegments)


def load_libero_actions_from_hdf5(hdf5_path):
    """
    Load all action sequences from a LIBERO HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        List of action arrays, each of shape (episode_length, action_dim)
    """
    all_actions = []

    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        for demo_key in data.keys():
            if demo_key.startswith('demo_'):
                actions = data[demo_key]['actions'][()]
                all_actions.append(actions)

    return all_actions


def load_all_libero_actions(data_dir, task_suite):
    """
    Load all action sequences from all HDF5 files in a LIBERO task suite.

    Args:
        data_dir: Root directory containing LIBERO datasets
        task_suite: Name of the task suite (e.g., 'libero_spatial_no_noops')

    Returns:
        List of action arrays
    """
    task_dir = os.path.join(data_dir, task_suite)

    if not os.path.exists(task_dir):
        raise ValueError(f"Task directory not found: {task_dir}")

    all_actions = []

    # Find all HDF5 files in the task directory
    hdf5_files = [f for f in os.listdir(task_dir) if f.endswith('.hdf5') or f.endswith('.h5')]

    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found in {task_dir}")

    print(f"Found {len(hdf5_files)} HDF5 files in {task_suite}")

    for hdf5_file in tqdm(hdf5_files, desc="Loading HDF5 files"):
        hdf5_path = os.path.join(task_dir, hdf5_file)
        actions = load_libero_actions_from_hdf5(hdf5_path)
        all_actions.extend(actions)

    print(f"Loaded {len(all_actions)} episodes total")

    return all_actions


def main(args):
    print(f"Training FAST tokenizer on LIBERO {args.task_suite} dataset")
    print(f"Parameters: T={args.T}, scale={args.scale}")

    # Load the base tokenizer from Hugging Face hub (official method)
    print("\nLoading base tokenizer from physical-intelligence/fast...")
    tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

    # Load all actions from LIBERO dataset
    print(f"\nLoading actions from {args.libero_data_dir}/{args.task_suite}...")
    all_actions = load_all_libero_actions(args.libero_data_dir, args.task_suite)

    # Split actions into subsegments
    print(f"\nSplitting actions into subsegments with T={args.T}...")
    all_subsegments = []
    for action in tqdm(all_actions, desc="Processing episodes"):
        subsegments = split_action_into_subsegments(action, args.T)
        if len(subsegments) > 0:
            all_subsegments.append(subsegments)

    if len(all_subsegments) == 0:
        raise ValueError("No valid subsegments generated. Check your data or T value.")

    all_subsegments = np.concatenate(all_subsegments, axis=0)
    print(f"Total subsegments: {all_subsegments.shape}")  # (num_subsegments, T, action_dim)

    # Test original tokenizer
    print("\n=== Testing original tokenizer ===")
    tokens = tokenizer(all_subsegments)
    decoded_actions = tokenizer.decode(tokens)

    # Compute reconstruction error
    diff = np.abs(all_subsegments - decoded_actions)
    mean_diff = np.mean(diff)
    print(f"Mean reconstruction error (before training): {mean_diff:.6f}")

    # Train the tokenizer
    print(f"\n=== Training tokenizer with scale={args.scale} ===")
    tokenizer = tokenizer.fit(all_subsegments, scale=args.scale)

    # Save the tokenizer
    os.makedirs(args.save_path, exist_ok=True)
    tokenizer.save_pretrained(args.save_path)
    print(f"Tokenizer saved to {args.save_path}")

    # Test trained tokenizer
    print("\n=== Testing trained tokenizer ===")
    tokens = tokenizer(all_subsegments)

    # Print token statistics
    token_lengths = [len(token) for token in tokens]
    print(f"Token length - mean: {np.mean(token_lengths):.2f}, max: {np.max(token_lengths)}, min: {np.min(token_lengths)}")

    # Compute reconstruction error
    decoded_actions = tokenizer.decode(tokens)
    mean_diff = np.mean(np.abs(all_subsegments - decoded_actions))
    print(f"Mean reconstruction error (after training): {mean_diff:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FAST action tokenizer on LIBERO dataset")

    parser.add_argument(
        "--libero_data_dir",
        type=str,
        default="/home/mike/ysz/MaskVLA/datasets/LIBERO_ORIGIN",
        help="Root directory containing LIBERO datasets"
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_90_no_noops",
        choices=["libero_spatial_no_noops", "libero_object_no_noops", "libero_goal_no_noops", "libero_10_no_noops", "libero_90_no_noops"],
        help="LIBERO task suite name"
    )
    parser.add_argument(
        "--T",
        type=int,
        default=8,
        help="Chunk size (subsegment length)"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=50,
        help="Scale parameter for tokenizer training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/home/mike/many_work/adapter-pro/fast-tokenizer/libero_90_no_noops",
        help="Path to save trained tokenizer"
    )

    args = parser.parse_args()
    main(args)
