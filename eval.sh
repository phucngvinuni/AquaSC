#!/bin/bash

echo "Starting SNR sweep evaluation script..."

SNR_LIST="-5 -3 -1 0 1 3 5 7 9 10 11 13 15 20"
CHECKPOINT_PATH="ckpt_semcom_reconstruction_yolo_fish_MIMO_LPIPS/checkpoint-429.pth"
CSV_OUTPUT_FILE="evaluation_results.csv"
MODEL_NAME="ViT_Reconstruction_Model_Default"
OUTPUT_DIR="evaluation_output"
DATA_PATH=""
YOLO_WEIGHTS="best.pt"
BATCH_SIZE=8
NUM_WORKERS=4
DEVICE="cuda"


BITS_VQ_HIGH=14
BITS_VQ_LOW=6
INPUT_SIZE=224
PATCH_SIZE=8
ENC_EMBED_DIM=512
ENC_DEPTH=8
ENC_NUM_HEADS=12

DEC_EMBED_DIM=512
DEC_DEPTH=6
DEC_NUM_HEADS=8
FIM_EMBED_DIM=256
FIM_DEPTH=2
FIM_NUM_HEADS=4


echo "SNR_dB,PSNR,SSIM,mAP_50,mAP" > "${CSV_OUTPUT_FILE}"
echo "Created results file: ${CSV_OUTPUT_FILE}"

for snr in $SNR_LIST
do
    echo ""
    echo "====================================================="
    echo "         RUNNING EVALUATION FOR SNR = ${snr} dB"
    echo "====================================================="

    python3 run_class_main.py \
        --model "${MODEL_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --data_set fish \
        --data_path "${DATA_PATH}" \
        --num_object_classes 1 \
        --yolo_weights "${YOLO_WEIGHTS}" \
        --batch_size ${BATCH_SIZE} \
        --input_size ${INPUT_SIZE} \
        --patch_size ${PATCH_SIZE} \
        --encoder_embed_dim ${ENC_EMBED_DIM} \
        --encoder_depth ${ENC_DEPTH} \
        --encoder_num_heads ${ENC_NUM_HEADS} \
        --decoder_embed_dim ${DEC_EMBED_DIM} \
        --decoder_depth ${DEC_DEPTH} \
        --decoder_num_heads ${DEC_NUM_HEADS} \
        --fim_embed_dim ${FIM_EMBED_DIM} \
        --fim_depth ${FIM_DEPTH} \
        --fim_num_heads ${FIM_NUM_HEADS} \
        --bits_vq_high ${BITS_VQ_HIGH} \
        --bits_vq_low ${BITS_VQ_LOW} \
        --num_workers ${NUM_WORKERS} \
        --device "${DEVICE}" \
        --resume "${CHECKPOINT_PATH}" \
        --eval \
        --snr_db_eval ${snr} \
        --yolo_conf_thres 0.4 \
        --yolo_iou_thres 0.5 \
        --csv_log_file "${CSV_OUTPUT_FILE}" \
        --append_csv

    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "  ERROR: Evaluation failed for SNR = ${snr} dB. Stopping script."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    fi
done

echo ""
echo "====================================================="
echo "      SNR SWEEP EVALUATION COMPLETE!"
echo "      Results saved to: ${CSV_OUTPUT_FILE}"
echo "====================================================="