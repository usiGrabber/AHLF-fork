#!/usr/bin/env python3
"""
Stream-based shuffle for large MGF files.

Strategy: External bucket shuffle
1. First pass: Stream through file, assign each spectrum to a random bucket (temp file)
2. Shuffle bucket order
3. Optionally shuffle within each bucket (if small enough)
4. Concatenate buckets to output

This keeps memory usage O(1) per spectrum while achieving good global shuffle.
"""

import os
import sys
import random
import tempfile
import shutil
import argparse
from pathlib import Path
from pyteomics import mgf
import gc


def write_spectrum_to_file(f, spectrum):
    """Write a single spectrum in MGF format."""
    f.write("BEGIN IONS\n")

    # Write parameters
    params = spectrum.get('params', {})
    for key, value in params.items():
        if isinstance(value, list):
            value = ' '.join(str(v) for v in value)
        f.write(f"{key.upper()}={value}\n")

    # Write peaks
    mz_array = spectrum['m/z array']
    intensity_array = spectrum['intensity array']
    for mz, intensity in zip(mz_array, intensity_array):
        f.write(f"{mz} {intensity}\n")

    f.write("END IONS\n\n")


def count_spectra_streaming(input_file):
    """Count spectra without loading entire file."""
    count = 0
    with mgf.read(input_file, use_index=False) as reader:
        for _ in reader:
            count += 1
            if count % 100000 == 0:
                print(f"  Counted {count:,} spectra...", flush=True)
    return count


def shuffle_mgf_external(input_file, output_file, num_buckets=100, seed=42,
                         shuffle_within_buckets=True, progress_interval=10000):
    """
    Shuffle a large MGF file using external bucket-based shuffling.

    Args:
        input_file: Path to input MGF file
        output_file: Path to output shuffled MGF file
        num_buckets: Number of temporary bucket files (more = better shuffle, more file handles)
        seed: Random seed for reproducibility
        shuffle_within_buckets: Whether to shuffle within each bucket (requires loading bucket into memory)
        progress_interval: Print progress every N spectra
    """
    random.seed(seed)

    input_path = Path(input_file)
    output_path = Path(output_file)

    print(f"=== Shuffling {input_path.name} ===", flush=True)
    print(f"  Input: {input_file}", flush=True)
    print(f"  Output: {output_file}", flush=True)
    print(f"  Buckets: {num_buckets}", flush=True)
    print(f"  Seed: {seed}", flush=True)

    # Create temp directory for buckets
    temp_dir = tempfile.mkdtemp(prefix="mgf_shuffle_")
    print(f"  Temp dir: {temp_dir}", flush=True)

    try:
        # --- PASS 1: Distribute spectra to random buckets ---
        print("\n[Pass 1] Distributing spectra to buckets...", flush=True)

        # Open all bucket files
        bucket_files = []
        bucket_counts = [0] * num_buckets
        for i in range(num_buckets):
            bucket_path = os.path.join(temp_dir, f"bucket_{i:04d}.mgf")
            bucket_files.append(open(bucket_path, 'w'))

        spectrum_count = 0
        with mgf.read(input_file, use_index=False) as reader:
            for spectrum in reader:
                # Skip empty spectra
                if len(spectrum['m/z array']) == 0:
                    continue

                # Assign to random bucket
                bucket_idx = random.randint(0, num_buckets - 1)
                write_spectrum_to_file(bucket_files[bucket_idx], spectrum)
                bucket_counts[bucket_idx] += 1
                spectrum_count += 1

                if spectrum_count % progress_interval == 0:
                    print(f"  Processed {spectrum_count:,} spectra...", flush=True)

                # Periodic cleanup
                if spectrum_count % 50000 == 0:
                    gc.collect()

        # Close all bucket files
        for f in bucket_files:
            f.close()

        print(f"  Total spectra: {spectrum_count:,}", flush=True)
        print(f"  Bucket sizes: min={min(bucket_counts):,}, max={max(bucket_counts):,}, "
              f"avg={sum(bucket_counts)//num_buckets:,}", flush=True)

        # --- PASS 2: Shuffle bucket order and concatenate ---
        print("\n[Pass 2] Shuffling and concatenating buckets...", flush=True)

        # Random bucket order
        bucket_order = list(range(num_buckets))
        random.shuffle(bucket_order)

        written_count = 0
        with open(output_file, 'w') as out_f:
            for i, bucket_idx in enumerate(bucket_order):
                bucket_path = os.path.join(temp_dir, f"bucket_{bucket_idx:04d}.mgf")

                if shuffle_within_buckets and bucket_counts[bucket_idx] > 0:
                    # Load bucket, shuffle, write
                    spectra = list(mgf.read(bucket_path, use_index=False))
                    random.shuffle(spectra)
                    for spectrum in spectra:
                        write_spectrum_to_file(out_f, spectrum)
                        written_count += 1
                    del spectra
                    gc.collect()
                else:
                    # Just copy bucket contents
                    with open(bucket_path, 'r') as bucket_f:
                        for line in bucket_f:
                            out_f.write(line)
                    written_count += bucket_counts[bucket_idx]

                if (i + 1) % 10 == 0:
                    print(f"  Merged {i + 1}/{num_buckets} buckets ({written_count:,} spectra)...", flush=True)

        print(f"  Written {written_count:,} spectra to output", flush=True)

    finally:
        # Cleanup temp directory
        print(f"\nCleaning up temp files...", flush=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Report output file size
    output_size = os.path.getsize(output_file) / (1024**3)
    print(f"\n=== Done! Output: {output_size:.2f} GB ===", flush=True)

    return spectrum_count


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle large MGF files using external bucket-based algorithm"
    )
    parser.add_argument("input_file", help="Input MGF file path")
    parser.add_argument("output_file", help="Output shuffled MGF file path")
    parser.add_argument("--buckets", type=int, default=100,
                        help="Number of temp buckets (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-shuffle-within", action="store_true",
                        help="Skip shuffling within buckets (faster but less random)")
    parser.add_argument("--progress", type=int, default=10000,
                        help="Progress report interval (default: 10000)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    shuffle_mgf_external(
        input_file=args.input_file,
        output_file=args.output_file,
        num_buckets=args.buckets,
        seed=args.seed,
        shuffle_within_buckets=not args.no_shuffle_within,
        progress_interval=args.progress
    )


if __name__ == "__main__":
    # If no arguments, shuffle the default files
    if len(sys.argv) == 1:
        base_dir = "/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/no_threshold"
        output_dir = "/sc/projects/sci-renard/usi-grabber/shared/mgf_files/final/no_threshold/shuffled"

        os.makedirs(output_dir, exist_ok=True)

        files_to_shuffle = [
            ("psm_data_with_phospho.phos.mgf", "psm_data_with_phospho.phos.shuffled.mgf"),
            ("psm_data_without_phospho.other.mgf", "psm_data_without_phospho.other.shuffled.mgf"),
        ]

        for input_name, output_name in files_to_shuffle:
            input_path = os.path.join(base_dir, input_name)
            output_path = os.path.join(output_dir, output_name)

            if os.path.exists(input_path):
                print(f"\n{'='*60}")
                shuffle_mgf_external(
                    input_file=input_path,
                    output_file=output_path,
                    num_buckets=200,  # More buckets for better shuffle on large files
                    seed=42,
                    shuffle_within_buckets=True
                )
            else:
                print(f"Skipping {input_name} - file not found")
    else:
        main()
