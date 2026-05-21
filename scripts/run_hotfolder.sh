#!/usr/bin/env bash
# Launch the image-to-splat hot folder daemon.
#
# Usage:
#   scripts/run_hotfolder.sh                    # defaults: 120v, 3072px, pixal3d, Three_Large_Tent
#   scripts/run_hotfolder.sh --num-views 200    # override any arg
#   tmux new -s splat scripts/run_hotfolder.sh  # keep running after SSH disconnect
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENCV_IO_ENABLE_OPENEXR=1

exec ~/miniconda3/envs/trellis2/bin/python -u scripts/hotfolder_daemon.py "$@"
