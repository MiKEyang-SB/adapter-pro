export CUDA_VISIBLE_DEVICES=0
setsid python experiments/robot/libero/run_libero_eval.py \
    --task_suite_name libero_object \
    --use_minivlm True \
    --pretrained_checkpoint outputs/configs+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--150005_chkpt \
    > eval.log 2>&1 &
    