export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export TORCH_DISTRIBUTED_DEBUG=DETAIL
setsid torchrun --standalone --nproc-per-node 1 vla-scripts/finetune.py \
    --resume=True \
    --resume_step=150005 \
    --resum_vla_path=/root/autodl-fs/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--150005_chkpt \
    --config_file_path=/root/autodl-fs/outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--150005_chkpt \
    --dataset_name='libero_object_no_noops' \
    --batch_size=16 \
    --max_steps=250005 \
    --learning_rate=2e-4 \
    --lr_warmup_ratio=0 \
    --use_minivlm=False \
    --num_steps_before_decay=250000 \
    --grad_accumulation_steps=1 \
    --data_root_dir='/root/autodl-fs/LIBERO_RLDS' \
    --run_root_dir='/root/autodl-fs/outputs' \
    --save_freq=50000 \
    --reset_scheduler_on_resume=True \
    > resume.log 2>&1 &