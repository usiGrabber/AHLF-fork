#!/usr/bin/env bash
set -euo pipefail

small="$1"   # file with reference lines
big="$2"     # large file to scan

# 1. Extract unique project prefixes from small file
awk -F: '{ print $1 ":" $2 }' "$small" | sort -u > /tmp/project_ids.txt

cat /tmp/project_ids.txt

echo "Starting overlap check"

# 2. Fast fixed-string substring match
grep -F -f /tmp/project_ids.txt "$big"
