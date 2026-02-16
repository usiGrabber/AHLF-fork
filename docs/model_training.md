# Model Training

## Data Pre-Processing
### Shuffling
Our goal is to create such a structure:

- training/0/with_phospho/*.mgf
- training/0/without_phospho/*mgf
- training/1/with_phospho/*.mgf
- training/1/without_phospho/*mgf

- val/with_phospho/data.mgf
- val/without_phospho/data.mgf


1. Split by project accession for phospho and other
If you have more data you might want to create more buckets. This will create two buckets 0/ and 1/
```bash
bash scripts/split_by_project.sh input_dir ouput_dir
```

2. You will need to move some files to recreate the structure from above

3. Shuffle into buckets
First check the individual sub folders sizes by running `du` inside the training folder. With our shuffle buffer of 100k we aim for roughly 20k spectra per file. We calculate the number of buckets per subfolder approximately with respect to the file size of the folder. On average one spectrum is roughly 14KiB. 

Let $N$ be the number of total spectra, $S_D$ the total file size of all spectra and $S_i$ the storage folder of a respective subfolder. The number of buckets for a subfolder will be set to:

$math.(S_i / (S_D / N)) / 20000$

We run the shuffle script for each subfolder with the correct bucket size and output dir:

```bash
INPUT_DIR="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/training/0/with_phospho/"
OUTPUT_DIR="/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/training_shuffled/0/with_phospho/"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# # Shuffle phospho file
echo ""
echo "=========================================="
echo "Shuffling PHOS file..."
echo "=========================================="
python /sc/home/konstantin.ketterer/AHLF-fork/shuffle_mgf.py \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    --buckets 109 \
    --seed 42 \
    --batch 10 \
    --progress 50000
```

### Validation Set Creation
We create a separate folder for validation and move ~400k spectra into that folder
```bash
mv training_shuffled/0/with_phospho/bucket_000[0-4]* validation/0/with_phospho/
mv training_shuffled/1/with_phospho/bucket_000[0-4]* validation/1/with_phospho/
mv training_shuffled/1/without_phospho/bucket_000[0-4]* validation/1/without_phospho/
mv training_shuffled/0/without_phospho/bucket_000[0-4]* validation/0/without_phospho/
```

### Random 50/50 Split Creation
Get all training files and split them randomly by their filename checksum

```bash
find "$(realpath training_shuffled)" -type f -name "*.mgf" | sed -E 's/\.(phos|other)\.mgf$//' > training_shuffled/trainings_files.txt
cat trainings_files.txt | bash ~/AHLF-fork/scripts/split_file_list.sh 
```

This results into two list of files that we will use for training two models.
