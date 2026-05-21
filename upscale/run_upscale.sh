#!/bin/bash
# WSL wrapper for AuraSR upscale folder-to-folder. Driven by the Windows BAT.
# Required args: $1 = input dir (WSL path), $2 = output dir (WSL path).

set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "usage: $0 <input_dir> <output_dir>"
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Set up conda + activate the env that has AuraSR + torch
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anysplat

# Ensure repo is on PYTHONPATH and run
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/upres.py" --input "$INPUT_DIR" --output "$OUTPUT_DIR"
