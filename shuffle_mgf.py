#!/usr/bin/env python3
"""
Stream-based shuffle for large MGF files using parallel bucket writing.

Strategy:
1. Stream through file, assign each spectrum to a random bucket
2. Use thread pool to write buckets in parallel
3. Output individual bucket files (no merge needed - training reads multiple files)

Output files are named like: bucket_0000.phos.mgf or bucket_0000.other.mgf
to match the glob patterns in dataset.py
"""

import os
import sys
import random
import argparse
from pathlib import Path
from pyteomics import mgf
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from queue import Queue
import gc


def format_spectrum(spectrum):
    """Format a single spectrum as MGF string."""
    lines = ["BEGIN IONS"]

    # Write parameters
    params = spectrum.get('params', {})
    for key, value in params.items():
        if isinstance(value, list):
            value = ' '.join(str(v) for v in value)
        lines.append(f"{key.upper()}={value}")

    # Write peaks
    mz_array = spectrum['m/z array']
    intensity_array = spectrum['intensity array']
    for mz, intensity in zip(mz_array, intensity_array):
        lines.append(f"{mz} {intensity}")

    lines.append("END IONS\n")
    return '\n'.join(lines)


class BucketWriter:
    """Thread-safe bucket writer with batched writes."""

    def __init__(self, output_dir, suffix, num_buckets, batch_size=100):
        self.output_dir = output_dir
        self.suffix = suffix  # e.g., ".phos.mgf" or ".other.mgf"
        self.num_buckets = num_buckets
        self.batch_size = batch_size

        # Per-bucket queues and locks
        self.queues = [[] for _ in range(num_buckets)]
        self.locks = [Lock() for _ in range(num_buckets)]
        self.counts = [0] * num_buckets

        # File handles (opened lazily)
        self.files = [None] * num_buckets
        self.file_locks = [Lock() for _ in range(num_buckets)]

        # Thread pool for parallel writes
        self.executor = ThreadPoolExecutor(max_workers=min(32, num_buckets))
        self.pending_futures = []

    def _get_file(self, bucket_idx):
        """Get or create file handle for bucket."""
        if self.files[bucket_idx] is None:
            path = os.path.join(self.output_dir, f"bucket_{bucket_idx:04d}{self.suffix}")
            self.files[bucket_idx] = open(path, 'w')
        return self.files[bucket_idx]

    def _flush_bucket(self, bucket_idx, spectra):
        """Write batch of spectra to bucket file."""
        with self.file_locks[bucket_idx]:
            f = self._get_file(bucket_idx)
            for spectrum_str in spectra:
                f.write(spectrum_str)

    def add_spectrum(self, bucket_idx, spectrum):
        """Add spectrum to bucket queue, flush if batch is full."""
        spectrum_str = format_spectrum(spectrum)

        with self.locks[bucket_idx]:
            self.queues[bucket_idx].append(spectrum_str)
            self.counts[bucket_idx] += 1

            if len(self.queues[bucket_idx]) >= self.batch_size:
                # Take the batch and clear queue
                batch = self.queues[bucket_idx]
                self.queues[bucket_idx] = []

                # Submit write to thread pool
                future = self.executor.submit(self._flush_bucket, bucket_idx, batch)
                self.pending_futures.append(future)

        # Periodic cleanup of completed futures
        if len(self.pending_futures) > 1000:
            self.pending_futures = [f for f in self.pending_futures if not f.done()]

    def close(self):
        """Flush remaining data and close all files."""
        print("  Flushing remaining buffers...", flush=True)

        # Flush all remaining queues
        for bucket_idx in range(self.num_buckets):
            with self.locks[bucket_idx]:
                if self.queues[bucket_idx]:
                    batch = self.queues[bucket_idx]
                    self.queues[bucket_idx] = []
                    future = self.executor.submit(self._flush_bucket, bucket_idx, batch)
                    self.pending_futures.append(future)

        # Wait for all writes to complete
        print("  Waiting for writes to complete...", flush=True)
        for future in self.pending_futures:
            future.result()

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        # Close all files
        for f in self.files:
            if f is not None:
                f.close()

        return self.counts


def shuffle_mgf_parallel(input_file, output_dir, num_buckets=100, seed=42,
                         batch_size=100, progress_interval=50000):
    """
    Shuffle a large MGF file using parallel bucket writing.

    Args:
        input_file: Path to input MGF file
        output_dir: Directory for output bucket files
        num_buckets: Number of bucket files to create
        seed: Random seed for reproducibility
        batch_size: Spectra to batch before writing
        progress_interval: Print progress every N spectra
    """
    random.seed(seed)

    input_path = Path(input_file)

    # Determine suffix from input file (.phos.mgf or .other.mgf)
    if '.phos.mgf' in input_path.name:
        suffix = '.phos.mgf'
    elif '.other.mgf' in input_path.name:
        suffix = '.other.mgf'
    else:
        suffix = '.mgf'

    print(f"=== Shuffling {input_path.name} ===", flush=True)
    print(f"  Input: {input_file}", flush=True)
    print(f"  Output dir: {output_dir}", flush=True)
    print(f"  Suffix: {suffix}", flush=True)
    print(f"  Buckets: {num_buckets}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    print(f"  Seed: {seed}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    # Create bucket writer
    writer = BucketWriter(output_dir, suffix, num_buckets, batch_size)

    try:
        print("\nDistributing spectra to buckets (parallel writes)...", flush=True)

        spectrum_count = 0
        with mgf.read(input_file, use_index=False) as reader:
            for spectrum in reader:
                # Skip empty spectra
                if len(spectrum['m/z array']) == 0:
                    continue

                # Assign to random bucket
                bucket_idx = random.randint(0, num_buckets - 1)
                writer.add_spectrum(bucket_idx, spectrum)
                spectrum_count += 1

                if spectrum_count % progress_interval == 0:
                    print(f"  Processed {spectrum_count:,} spectra...", flush=True)

                # Periodic memory cleanup
                if spectrum_count % 100000 == 0:
                    gc.collect()

        print(f"\n  Total spectra read: {spectrum_count:,}", flush=True)

    finally:
        bucket_counts = writer.close()

    print(f"  Bucket sizes: min={min(bucket_counts):,}, max={max(bucket_counts):,}, "
          f"avg={sum(bucket_counts)//num_buckets:,}", flush=True)

    # Count non-empty buckets
    non_empty = sum(1 for c in bucket_counts if c > 0)
    print(f"  Non-empty buckets: {non_empty}/{num_buckets}", flush=True)

    print(f"\n=== Done! Output files: {output_dir}/bucket_*{suffix} ===", flush=True)

    return spectrum_count


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle large MGF files using parallel bucket writing"
    )
    parser.add_argument("input_file", help="Input MGF file path")
    parser.add_argument("output_dir", help="Output directory for bucket files")
    parser.add_argument("--buckets", type=int, default=100,
                        help="Number of bucket files (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--batch", type=int, default=100,
                        help="Batch size for writes (default: 100)")
    parser.add_argument("--progress", type=int, default=50000,
                        help="Progress report interval (default: 50000)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    shuffle_mgf_parallel(
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_buckets=args.buckets,
        seed=args.seed,
        batch_size=args.batch,
        progress_interval=args.progress
    )


if __name__ == "__main__":
    main()
