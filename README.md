# AquaSC: Task-Oriented Semantic Communication for Efficient and Intelligent Aquaculture IoT Systems

[![paper-link-placeholder](https://img.shields.io/badge/Paper-Link%20to%20be%20added-blue)](https://#) <!-- Add link to your paper here -->

This repository contains the official PyTorch implementation for the paper **"AquaSC: Task-Oriented Semantic Communication for Efficient and Intelligent Aquaculture IoT Systems"**.

AquaSC is a novel task-oriented semantic communication (SC) framework designed to address the critical bandwidth constraints in smart aquaculture. By intelligently prioritizing information relevant to fish detection, AquaSC significantly reduces data transmission volume while maintaining high performance on the downstream task.

 <!-- Replace with a link to your model architecture diagram -->
*Fig. 1: The end-to-end digital semantic communication pipeline of AquaSC.*

### Key Features
- **Task-Oriented Compression**: The system is optimized end-to-end not for pixel-perfect reconstruction, but for maximizing the accuracy of a downstream fish detection model (YOLO).
- **Adaptive Bit Allocation**: A **Feature Importance Module (FIM)** identifies semantically important image patches (i.e., those containing fish) and routes them to a high-fidelity quantizer, while aggressively compressing the background with a low-fidelity one.
- **Task-Aware Learning**: A **Bounding Box Weighted (BBW)** reconstruction loss prioritizes image quality within object regions during training, directly steering the model to learn representations useful for detection.
- **VinFish Dataset**: We introduce and will make publicly available the **VinFish** dataset, a new collection of annotated underwater images from aquaculture environments, to facilitate research in this domain.

---

## 1. Setup

### 1.1. Clone the Repository
```bash
git clone https://github.com/your-username/AquaSC.git
cd AquaSC
```

### 1.2. Create Conda Environment
We provide an `environment.yml` file to ensure all dependencies are correctly installed. This setup is tested on a system with NVIDIA GPUs and CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate aquasc-env
```

### 1.3. Prepare the Dataset
1.  Download the **VinFish** dataset from [this link](https://drive.google.com/file/d/1S4wye_MOeDrs8uBzcMSN15ChbvCbH77Z/view?usp=sharing) <!-- Add dataset download link --> and unzip it.
2.  Arrange the dataset into the following structure:
    ```
    /path/to/your/dataset/
    ├── train/
    │   ├── images/
    │   │   ├── 0001.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── 0001.txt
    │       └── ...
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    ```
3.  Download a pre-trained YOLO model (e.g., YOLOv8s) for the downstream task evaluation and place the weight file (`best.pt` or `yolov8s.pt`) in the root directory of this project.

---

## 2. Training
The main training script is `run_training.py`. We provide a convenience script `train.sh` to start the training process.

Before running, **edit `train.sh`** to set the correct paths:
- `DATA_PATH`: Path to the root of your prepared dataset.
- `YOLO_WEIGHTS`: Path to your pre-trained YOLO weight file.
- `OUTPUT_DIR`: Directory where checkpoints and logs will be saved.

```bash
#!/bin/bash
# train.sh

DATA_PATH="/path/to/your/dataset/"
YOLO_WEIGHTS="yolov8s.pt"
OUTPUT_DIR="checkpoints/aquasc_b16_4"

CUDA_VISIBLE_DEVICES=0 python3 run_training.py \
    --model ViT_Reconstruction_Model_Default \
    --output_dir "${OUTPUT_DIR}" \
    --data_set fish \
    --data_path "${DATA_PATH}" \
    --yolo_weights "${YOLO_WEIGHTS}" \
    --batch_size 2 \
    --input_size 224 \
    --patch_size 8 \
    --encoder_embed_dim 512 \
    --encoder_depth 8 \
    --encoder_num_heads 12 \
    --decoder_embed_dim 512 \
    --decoder_depth 6 \
    --decoder_num_heads 8 \
    --quantizer_dim 512 \
    --bits_vq_high 16 \
    --bits_vq_low 4 \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --warmup_epochs 20 \
    --epochs 300 \
    --opt adamw \
    --weight_decay 0.05 \
    --clip_grad 1.0 \
    --save_freq 10 \
    --save_ckpt \
    --snr_db_train_min 0 \
    --snr_db_train_max 20 \
    --snr_db_eval 10 \
    --num_workers 4 \
    --inside_box_loss_weight 10 \
    --outside_box_loss_weight 0.1
```

Once configured, run the training:
```bash
bash train.sh
```
Checkpoints will be saved periodically in the specified `OUTPUT_DIR`, including `checkpoint-best_psnr.pth` and `checkpoint-best_map.pth`.

---

## 3. Evaluation
To evaluate a trained model, use the `evaluate.sh` script. This script iterates through a list of SNR values, runs the evaluation, and logs the results to a CSV file.

**Edit `evaluate.sh`** to set the correct paths:
- `CHECKPOINT_PATH`: Path to your trained AquaSC model checkpoint (e.g., `checkpoint-best_map.pth`).
- `DATA_PATH` and `YOLO_WEIGHTS` should also be set correctly.

```bash
#!/bin/bash
# evaluate.sh

# --- Configuration ---
CHECKPOINT_PATH="checkpoints/aquasc_b16_4/checkpoint-best_map.pth"
DATA_PATH="/path/to/your/dataset/"
YOLO_WEIGHTS="yolov8s.pt"
OUTPUT_CSV="results/evaluation_results.csv"

# List of SNR values (in dB) to evaluate on
SNR_LIST=(0 5 10 15 20 25)

# --- Do not edit below ---
# ... (script logic) ...
```

Run the evaluation script:
```bash
bash evaluate.sh
```
The script will print the results for each SNR to the console and append them to the specified CSV file, creating a summary table of the model's performance across different channel conditions.

---

## 4. Reconstructing a Dataset
You can use a trained AquaSC model to transmit and reconstruct an entire dataset split. This is useful for visualizing model performance or for training a downstream model on the reconstructed data.

**Edit `reconstruct.sh`** to configure the paths and parameters:
- `CHECKPOINT_PATH`: Path to your trained AquaSC model.
- `ORIGINAL_DATA_ROOT`: Path to the original dataset.
- `RECON_DATA_ROOT`: Path where the new, reconstructed dataset will be created.
- `SPLIT_TO_RECONSTRUCT`: The split to process (`train`, `valid`, or `test`).
- `SNR_FOR_RECONSTRUCTION`: The channel SNR (dB) to simulate.

```bash
bash reconstruct.sh
```
This will create a new directory structure under `RECON_DATA_ROOT` containing the reconstructed images and copied labels.

---

## Citation
If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{nguyen2024aquasc,
  title={{AquaSC}: Task-Oriented Semantic Communication for Efficient and Intelligent Aquaculture {IoT} Systems},
  author={Phuc H. Nguyen and Trung T. Nguyen and Senura Hansaja Wanasekara and Ngoc M. Ngo and Van-Dinh Nguyen},
  journal={Journal of \LaTeX\ Class Files},
  volume={14},
  number={8},
  year={2024},
  publisher={IEEE}
}
```
