#!/bin/bash
# WSL wrapper for the AutoCrop tool. Operates on $IMAGE2SPLAT_HOTFOLDER/inbox by
# default — the same folder the daemon watches.
#
# Usage:
#   bash run_autocrop.sh              # crops $IMAGE2SPLAT_HOTFOLDER/inbox
#   bash run_autocrop.sh /some/path   # crops that folder
#   bash run_autocrop.sh /path 60     # crops that folder with 60px margin

set -e

# Resolve the target folder
if [ -n "$1" ]; then
    INPUT_DIR="$1"
elif [ -n "$IMAGE2SPLAT_HOTFOLDER" ]; then
    INPUT_DIR="$IMAGE2SPLAT_HOTFOLDER/inbox"
else
    INPUT_DIR="$HOME/image2splat/inbox"
fi

MARGIN="${2:-40}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "FAIL: input dir doesn't exist: $INPUT_DIR"
    exit 1
fi

# Activate the daemon's conda env (numpy + Pillow are enough — no GPU needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate trellis2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/autocrop.py" --input "$INPUT_DIR" --margin "$MARGIN"
