# AHLF - usiGrabber Fork

This repository is a fork based on [AHLF](https://gitlab.com/dacs-hpi/AHLF)  and contains the code used to train an AHLF style model based on the dataset collected by the [usiGrabber](https://github.com/usiGrabber/usiGrabber).

- We have published our final model weights here: [model/usigrabber_model_weights.hdf5](/model/usigrabber_model_weights.hdf5).

***AHLF: ad hoc learning of peptide fragmentation from mass spectra enables an interpretable detection of phosphorylated and cross-linked peptides***

Tom Altenburg, Sven Giese, Shengbo Wang, Thilo Muth, Bernhard Y. Renard  
bioRxiv 2020.05.19.101345; doi: https://doi.org/10.1101/2020.05.19.101345 

## Getting started

To install the required packages create a new conda environment (required packages are automatically installed when using **`ahlf_training_env.yml`**):

```
conda env create -f ahlf_training_env.yml
conda activate ahlf_training_env
```

## Preproccesing

Because our usiGrabber download tool produces one MGF file per original mzIdentML file, we must first shuffle this data. We provide the following helper scripts for it: [shuffle_mgf.py](/shuffle_mgf.py), [shuffle_mgf_training.slurm](/shuffle_mgf_training.slurm). 

You should move any data used for validation into a separate folder after shuffling.

## Training

In order to train the model, e.g. on user-specific data, you can use **`training.py`**:

Hyperparameters and the dataset location must be changed in the **`training.py`** script directly. We provide a [helper script](/training.slurm) to run training on a SLURM cluster.

## Inference & Evaluation

Inference and evalution can be made with the script [inference-test.slurm](/inference-test.slurm). You must adjust `MODEL_DIR`, `MODEL_NAME` and provide two class MGF files for your testing data (`PHOSPHO_MGF`, `NON_PHOSPHO_MGF`).

Note: Our model uses a different ion current normalization function. If you evaluate the original AHLF checkpoints you must change it back to `NAME=ORIGINAL` in [dataset.py](/dataset.py).


### Disclaimer 
Any other script in this repository are not maintained and guranteed to work. Please refer to the original [AHLF repository](https://gitlab.com/dacs-hpi/AHLF).

