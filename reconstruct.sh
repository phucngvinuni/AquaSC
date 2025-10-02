#!/bin/bash

echo "Starting single image inference script..."

PROJECT_ROOT="/mnt/c/Users/ADMIN/Downloads/siso" 
CHECKPOINT_FOLDER_NAME="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS"
CHECKPOINT_FILE_NAME="checkpoint-399.pth"

INPUT_IMAGE_PATH="/mnt/c/Users/ADMIN/Downloads/siso/ori.png"
OUTPUT_IMAGE_PATH="./reconstructed_output_image 164 snr 0.png"

MODEL_ARCH_NAME="ViT_Reconstruction_Model_Default"
IMG_INPUT_SIZE=224
VIT_PATCH_SIZE=8

ENC_EMBED_DIM_CKPT=512
ENC_DEPTH_CKPT=8
ENC_HEADS_CKPT=12

DEC_EMBED_DIM_CKPT=512
DEC_DEPTH_CKPT=6
DEC_HEADS_CKPT=8

QUANTIZER_DIM_CKPT=512
BITS_VQ_HIGH_CKPT=16 
BITS_VQ_LOW_CKPT=4    

FIM_EMBED_DIM_CKPT=256
FIM_DEPTH_CKPT=2
FIM_HEADS_CKPT=4
FIM_ROUTING_THRESHOLD_CKPT=0.6

DROP_RATE_CKPT=0.0
DROP_PATH_RATE_CKPT=0.1
# ---------------------------------------------------------------------------

# --- INFERENCE PARAMETERS ---
SNR_TO_EVALUATE=0 
DEVICE_TO_USE="cuda"
SEED_VALUE=42

FULL_CHECKPOINT_PATH="${PROJECT_ROOT}/${CHECKPOINT_FOLDER_NAME}/${CHECKPOINT_FILE_NAME}"

python3 "${PROJECT_ROOT}/reconimg.py" \
    --semcom_checkpoint_path "${FULL_CHECKPOINT_PATH}" \
    --input_image_path "${INPUT_IMAGE_PATH}" \
    --output_image_path "${OUTPUT_IMAGE_PATH}" \
    \
    --model "${MODEL_ARCH_NAME}" \
    --input_size ${IMG_INPUT_SIZE} \
    --patch_size ${VIT_PATCH_SIZE} \
    \
    --encoder_embed_dim ${ENC_EMBED_DIM_CKPT} \
    --encoder_depth ${ENC_DEPTH_CKPT} \
    --encoder_num_heads ${ENC_HEADS_CKPT} \
    \
    --decoder_embed_dim ${DEC_EMBED_DIM_CKPT} \
    --decoder_depth ${DEC_DEPTH_CKPT} \
    --decoder_num_heads ${DEC_HEADS_CKPT} \
    \
    --quantizer_dim ${QUANTIZER_DIM_CKPT} \
    --bits_vq_high ${BITS_VQ_HIGH_CKPT} \
    --bits_vq_low ${BITS_VQ_LOW_CKPT} \
    \
    --fim_embed_dim ${FIM_EMBED_DIM_CKPT} \
    --fim_depth ${FIM_DEPTH_CKPT} \
    --fim_num_heads ${FIM_HEADS_CKPT} \
    --fim_routing_threshold ${FIM_ROUTING_THRESHOLD_CKPT} \
    \
    --drop_rate ${DROP_RATE_CKPT} \
    --drop_path_rate ${DROP_PATH_RATE_CKPT} \
    \
    --snr_eval ${SNR_TO_EVALUATE} \
    --device "${DEVICE_TO_USE}" \
    --seed ${SEED_VALUE}

echo "Single image inference script finished."