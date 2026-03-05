#!/bin/bash
#SBATCH --job-name=ahlf-ensemble-inference-2
#SBATCH --output=slurm_ensemble_inference_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=gpu-batch
#SBATCH --gpus=1
#SBATCH --constraint="GPU_SKU:V100&ARCH:X86"
#SBATCH --account=sci-renard-usi-grabber

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ahlf_interpretation_env

# Define input files
PHOSPHO_MGF="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/validation_sp_p.mgf"
NON_PHOSPHO_MGF="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/validation_sp_non_p.mgf"

# Create output directory
mkdir -p ensemble_results

echo "Running inference with all four models..."


MODEL_DIR=checkpoints
MODEL_NAME="ADD YOUR MODEL NAME WITHOUT THE EXTENSION RIGHT HERE"

RESULTS_DIR=ensemble_results/$MODEL_NAME

mkdir -p $RESULTS_DIR

# Run alpha model
echo "Running alpha model on phospho spectra..."
python inference.py $MODEL_DIR/$MODEL_NAME.hdf5 $PHOSPHO_MGF $RESULTS_DIR/phospho.tsv --tsv

echo "Running alpha model on non-phospho spectra..."
python inference.py $MODEL_DIR/$MODEL_NAME.hdf5 $NON_PHOSPHO_MGF $RESULTS_DIR/non_phospho.tsv --tsv

echo "All inference runs complete!"
echo "Running ensemble combination and evaluation..."

# Run the combination script
python combine_ensemble_results.py $RESULTS_DIR --output $RESULTS_DIR/ensemble_combined_results.tsv

echo "Done!"