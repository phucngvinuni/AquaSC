#!/bin/bash

# This sets the GPU to be used (e.g., GPU 0)
export CUDA_VISIBLE_DEVICES=0

# This is the command to run your Python training script
python3 run_class_main.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS \
    --data_set fish \
    --data_path "" \
    --num_object_classes 1 \
    --yolo_weights "best.pt" \
    --batch_size 2 \
    --input_size 224 \
    --encoder_embed_dim 512 \
    --encoder_depth 8 \
    --encoder_num_heads 12 \
    --patch_size 8 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_epochs 20 \
    --epochs 1000 \
    --opt adamw \
    --weight_decay 0.05 \
    --clip_grad 1.0 \
    --save_freq 10 \
    --save_ckpt \
    --mask_ratio 0.0 \
    --snr_db_train_min 0 \
    --snr_db_train_max 20 \
    --snr_db_eval 10 \
    --num_workers 4 \
    --fim_routing_threshold 0.6 \
    --bits_vq_high 16 \
    --bits_vq_low 4 \
    --fim_loss_weight 0.5 \
    --vq_loss_weight 0.15 \
    --lpips_loss_weight 0 \
    --yolo_conf_thres 0.4 \
    --yolo_iou_thres 0.5 \
    --inside_box_loss_weight 10 \
    --outside_box_loss_weight 0.1 \
    # --resume "ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS/checkpoint-649.pth" \
    # --eval \
