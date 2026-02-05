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

# Input/output paths
INPUT_DIR="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/no_threshold"
OUTPUT_DIR="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/no_threshold_shuffled3"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# # Shuffle phospho file
echo ""
echo "=========================================="
echo "Shuffling PHOS file..."
echo "=========================================="
python /sc/home/konstantin.ketterer/AHLF-fork/shuffle_mgf.py \
    "${INPUT_DIR}/psm_data_with_phospho.phos.mgf" \
    "${OUTPUT_DIR}" \
    --buckets 400 \
    --seed 42 \
    --batch 500 \
    --progress 50000

# # Shuffle other file
# echo ""
# echo "=========================================="
# echo "Shuffling OTHER file..."
# echo "=========================================="
# python /sc/home/konstantin.ketterer/AHLF-fork/shuffle_mgf.py \
#     "${INPUT_DIR}/psm_data_without_phospho.other.mgf" \
#     "${OUTPUT_DIR}" \
#     --buckets 400 \
#     --seed 42 \
#     --batch 500 \
#     --progress 50000

# echo ""
echo "Finished at $(date)"
