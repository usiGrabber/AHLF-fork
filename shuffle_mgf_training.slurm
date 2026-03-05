#!/bin/bash
#SBATCH --job-name=shuffle_mgf
#SBATCH --output=slurm_shuffle_mgf_%j.out
#SBATCH --error=slurm_shuffle_mgf_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu-batch
#SBATCH --account=sci-renard-usi-grabber

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ahlf_training_env

echo "Starting MGF shuffle job at $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"

# Input/output paths, script path!
INPUT_DIR="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/training"
OUTPUT_DIR="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/training_shuffled_final"
SCRIPT_PATH="path to shuffle_mgf.py"

PARTITION="0/with_phospho/"
# Create output directory
mkdir -p "${OUTPUT_DIR}/${PARTITION}"

# # Shuffle phospho file
echo ""
echo "=========================================="
echo "Shuffling file..."
echo "=========================================="
python ${SCRIPT_PATH} \
    "${INPUT_DIR}/${PARTITION}" \
    "${OUTPUT_DIR}/${PARTITION}" \
    --buckets 420 \
    --seed 542 \
    --batch 10 \
    --progress 50000

PARTITION="0/without_phospho/"
mkdir -p "${OUTPUT_DIR}/${PARTITION}"
python ${SCRIPT_PATH} \
    "${INPUT_DIR}/${PARTITION}" \
    "${OUTPUT_DIR}/${PARTITION}" \
    --buckets 730 \
    --seed 532 \
    --batch 10 \
    --progress 50000

PARTITION="1/with_phospho/"
mkdir -p "${OUTPUT_DIR}/${PARTITION}"
python ${SCRIPT_PATH} \
    "${INPUT_DIR}/${PARTITION}" \
    "${OUTPUT_DIR}/${PARTITION}" \
    --buckets 477 \
    --seed 4855 \
    --batch 10 \
    --progress 50000

PARTITION="1/without_phospho/"
mkdir -p "${OUTPUT_DIR}/${PARTITION}"
python ${SCRIPT_PATH} \
    "${INPUT_DIR}/${PARTITION}" \
    "${OUTPUT_DIR}/${PARTITION}" \
    --buckets 370 \
    --seed 5163 \
    --batch 10 \
    --progress 50000

echo "Finished at $(date)"
