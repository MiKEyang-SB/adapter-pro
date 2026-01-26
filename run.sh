export CUDA_VISIBLE_DEVICES=1,2,3
export HF_ENDPOINT=https://hf-mirror.com
setsid torchrun --standalone --nproc-per-node 3 vla-scripts/finetune.py \
    --data_root_dir=data \
    --dataset_name='libero_object_no_noops' \
    --batch_size=12 \
    --learning_rate=3e-4 \
    --max_steps=100005 \
    --lr_warmup_ratio=0.05 \
    --num_steps_before_decay=100000 \
    --grad_accumulation_steps=1 \
    > train.log 2>&1 &