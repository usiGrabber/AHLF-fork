#!/usr/bin/env bash
set -euo pipefail

small="$1"
big="$2"

awk '
  NR==FNR { a[$0]=1; next }   # load small file into hash
  ($0 in a) { print; found=1 }
  END { exit(found ? 0 : 1) }
' "$small" "$big"
