#!/usr/bin/env bash
set -euo pipefail

small="$1"
big="$2"

# Extract unique "file name" part (3rd colon-separated field)
awk -F: '{ print $3 }' "$small" | sort -u > /tmp/filenames.txt

head /tmp/filenames.txt
wc -l /tmp/filenames.txt
echo "Starting overlap search"

# Fixed-string substring search
grep -F -f /tmp/filenames.txt "$big"
