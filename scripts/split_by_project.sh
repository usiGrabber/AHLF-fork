#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <BASE_DIR> <TARGET_DIR>" >&2
    exit 1
fi

BASE_DIR="$1"
TARGET_DIR="$2"

bucket0=0
bucket1=0

mkdir -p "$TARGET_DIR/0" "$TARGET_DIR/1"

# If no files match, don't iterate over literal '*'
shopt -s nullglob

for f in "$BASE_DIR"/*; do
    [[ -f "$f" ]] || continue

    fname="$(basename "$f")"

    # part before first underscore
    prefix="${fname%%_*}"

    # file size in bytes
    size="$(stat -c '%s' "$f")"

    hash="$(printf '%s' "$prefix" | cksum | awk '{print $1}')"
    bucket=$(( hash % 2 ))

    echo "$f → bucket $bucket (prefix=$prefix)"
    echo "  copying to: $TARGET_DIR/$bucket/$fname"

    if [[ "$bucket" -eq 0 ]]; then
        bucket0=$((bucket0 + size))
    else
        bucket1=$((bucket1 + size))
    fi

    cp -- "$f" "$TARGET_DIR/$bucket/$fname"
done

echo
echo "Bucket 0 total size: $bucket0 bytes"
echo "Bucket 1 total size: $bucket1 bytes"
