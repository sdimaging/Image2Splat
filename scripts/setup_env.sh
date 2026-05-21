#!/usr/bin/env bash
# Bootstrap the TRELLIS.2 environment for the image_to_splat pipeline.
#
# What this script does:
#   1. Sanity check: WSL2 Ubuntu, GPU passthrough, free disk, no /mnt/c install path
#   2. Install WSL-specific CUDA 12.8 toolkit (requires sudo) — Blackwell support
#   3. Install system packages required by nvdiffrast/EGL
#   4. Clone microsoft/TRELLIS.2 into ~/projects/TRELLIS.2/
#   5. Run TRELLIS.2's setup.sh to build the conda env (30-90 min)
#
# Re-run safe: each step checks for prior completion before acting.

set -euo pipefail

PROJECTS_DIR="$HOME/projects"
TRELLIS_DIR="$PROJECTS_DIR/TRELLIS.2"
MIN_FREE_GB=120  # 85GB models + repos + build artifacts + slack

color() { printf "\033[1;%sm%s\033[0m\n" "$1" "$2"; }
log()   { color 36 "==> $*"; }
ok()    { color 32 "  ✓ $*"; }
warn()  { color 33 "  ! $*"; }
fail()  { color 31 "  ✗ $*"; exit 1; }

# ---- 1. sanity checks ------------------------------------------------------

log "Step 1/6: sanity checks"

if ! grep -qi microsoft /proc/version; then
    fail "not running inside WSL2 — this script assumes WSL2 Ubuntu"
fi
ok "WSL2 detected"

if ! command -v nvidia-smi >/dev/null; then
    fail "nvidia-smi not found — GPU passthrough is not configured"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "GPU passthrough OK: $GPU_NAME"

FREE_GB=$(df -BG "$HOME" | awk 'NR==2 {gsub("G","",$4); print $4}')
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    fail "only ${FREE_GB}G free in WSL native FS; need at least ${MIN_FREE_GB}G"
fi
ok "free disk in WSL native FS: ${FREE_GB}G"

SCRIPT_PATH=$(readlink -f "$0")
case "$SCRIPT_PATH" in
    /mnt/*) fail "this project is on a Windows-mounted path ($SCRIPT_PATH). Move it to ~/projects/ and re-run." ;;
esac
ok "project lives on Linux-native filesystem"

# ---- 2. CUDA 12.8 toolkit (WSL-specific, Blackwell-capable) ----------------
#
# CUDA 12.8 is the first toolkit version that supports `compute_120` codegen
# (Blackwell / RTX 50-series). Torch 2.7.1+cu128 builds extensions targeting
# compute_120; nvcc < 12.8 will refuse with `Unsupported gpu architecture`.

log "Step 2/6: CUDA 12.8 toolkit (Blackwell sm_120 codegen)"

# Detect existing 12.8 install via the canonical path (don't rely on $PATH).
if [ -x /usr/local/cuda-12.8/bin/nvcc ] && /usr/local/cuda-12.8/bin/nvcc --version | grep -q "release 12.8"; then
    ok "CUDA 12.8 toolkit already installed"
else
    warn "installing cuda-toolkit-12-8 from NVIDIA's wsl-ubuntu repo (requires sudo)"
    cd /tmp
    if [ ! -f cuda-keyring_1.1-1_all.deb ]; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    fi
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    # Slim subset — skip nsight-systems et al. (libtinfo5 conflicts on Noble +
    # we don't use profilers anyway). Just the compiler + dev headers/libs that
    # flash-attn / nvdiffrast / nvdiffrec / cumesh / FlexGEMM / o-voxel need.
    sudo apt-get -y install \
        cuda-toolkit-12-8-config-common \
        cuda-compiler-12-8 \
        cuda-nvcc-12-8 \
        cuda-cudart-dev-12-8 \
        cuda-nvrtc-dev-12-8 \
        cuda-driver-dev-12-8 \
        cuda-libraries-dev-12-8 \
        cuda-profiler-api-12-8 \
        cuda-cccl-12-8
    ok "CUDA 12.8 toolkit (slim — no nsight/profiler) installed"
fi

# Ensure CUDA_HOME is in shell rc. Strip any older cuda-12.4 block first.
if grep -q "CUDA_HOME=/usr/local/cuda-12.4" "$HOME/.bashrc"; then
    warn "removing old CUDA 12.4 block from ~/.bashrc (upgrading to 12.8)"
    # Remove the 4-line block we previously appended for 12.4
    sed -i '/# image_to_splat \/ TRELLIS.2 — CUDA 12.4/,/^export LD_LIBRARY_PATH=\$CUDA_HOME\/lib64/d' "$HOME/.bashrc"
fi
if ! grep -q "CUDA_HOME=/usr/local/cuda-12.8" "$HOME/.bashrc"; then
    cat >> "$HOME/.bashrc" <<'EOF'

# image_to_splat / TRELLIS.2 — CUDA 12.8 (WSL-specific toolkit, Blackwell-capable)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
EOF
    ok "CUDA 12.8 env vars appended to ~/.bashrc"
else
    ok "CUDA 12.8 env vars already in ~/.bashrc"
fi
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
# LIBRARY_PATH = linker's compile-time search path. nvdiffrec/cumesh/FlexGEMM/o-voxel
# link against -lcuda (the CUDA *driver* lib). On WSL2 the real driver lib is at
# /usr/lib/wsl/lib/libcuda.so.1, but the toolkit ships a stub libcuda.so for
# compile-time linking. Without this export the extensions fail with
# "/usr/bin/ld: cannot find -lcuda".
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}

# ---- 3. system packages ----------------------------------------------------

log "Step 3/6: system packages (build tools + EGL for nvdiffrast)"

sudo apt-get install -y \
    build-essential git curl wget \
    libopencv-dev libegl1-mesa-dev libgl1-mesa-dev \
    libglib2.0-0 ffmpeg
ok "system packages installed"

# ---- 4. clone TRELLIS.2 ----------------------------------------------------

log "Step 4/6: clone microsoft/TRELLIS.2"

mkdir -p "$PROJECTS_DIR"
if [ -d "$TRELLIS_DIR/.git" ]; then
    ok "TRELLIS.2 already cloned at $TRELLIS_DIR"
else
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive "$TRELLIS_DIR"
    ok "cloned to $TRELLIS_DIR"
fi

# ---- 5. build the trellis2 conda env ---------------------------------------

log "Step 5/6: build trellis2 conda env (30-90 min — extensions compile from source)"

# Pre-install libjpeg-dev so setup.sh's BASIC block doesn't need a fresh sudo
sudo apt-get install -y libjpeg-dev

# Source conda's shell hook into THIS bash session so `conda activate` works.
# Non-interactive shells don't run ~/.bashrc, so the hook isn't auto-loaded.
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Patch TRELLIS.2's setup.sh to use the Blackwell-capable stack.
# TRELLIS.2 upstream pins torch==2.6.0+cu124, whose wheels were built without
# sm_120 (Blackwell) → matmul fails on RTX 5090. We pin to torch 2.7.1+cu128
# instead, which has sm_120 SASS in the wheel (verified by torch.cuda.get_arch_list()).
# Also bump flash-attn to 2.8.2 which has matching prebuilt wheels for torch 2.7+cu12.
# These sed patches are idempotent — no-op if already patched.
log "patching TRELLIS.2 setup.sh for Blackwell (torch 2.7.1+cu128, flash-attn 2.8.2)"
sed -i \
    -e 's|pip install torch==2\.6\.0 torchvision==0\.21\.0 --index-url https://download\.pytorch\.org/whl/cu124|pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128|' \
    -e 's|pip install flash-attn==2\.7\.3|pip install flash-attn==2.8.2|' \
    "$TRELLIS_DIR/setup.sh"

# Pinned target stack
TORCH_TARGET_VER="2.7.1+cu128"
FLASH_ATTN_VER="2.8.2"
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/flash_attn-${FLASH_ATTN_VER}%2Bcu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

TRELLIS_FLAGS=""
NEED_NUKE=false
if conda env list | awk '{print $1}' | grep -qx 'trellis2'; then
    TORCH_VER=$(conda run -n trellis2 python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
    case "$TORCH_VER" in
        $TORCH_TARGET_VER*|2.7.1*cu128*)
            ok "trellis2 env has correct torch ($TORCH_VER) — resuming install"
            conda activate trellis2
            TRELLIS_FLAGS="--basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"
            ;;
        missing)
            warn "trellis2 env exists but torch not installed — nuking + rebuilding cleanly"
            NEED_NUKE=true
            ;;
        *)
            warn "trellis2 env has WRONG torch ($TORCH_VER); target is $TORCH_TARGET_VER — nuking + rebuilding"
            NEED_NUKE=true
            ;;
    esac
else
    TRELLIS_FLAGS="--new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"
fi

if [ "$NEED_NUKE" = true ]; then
    conda env remove -n trellis2 -y
    TRELLIS_FLAGS="--new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"
fi

if [ -n "$TRELLIS_FLAGS" ]; then
    # setup.sh git-clones each extension to /tmp/extensions/<name>, which fails
    # if a previous partial run already left dirs there. Clear them so retries
    # work cleanly. Idempotent — does nothing if dir is empty/absent.
    if [ -d /tmp/extensions ]; then
        log "clearing /tmp/extensions/ from any previous partial run"
        rm -rf /tmp/extensions
    fi

    cd "$TRELLIS_DIR"

    # If we're doing a fresh env build, setup.sh's --new-env block installs torch.
    # We then need flash-attn installed BEFORE setup.sh hits its `pip install flash-attn==2.7.3`
    # line, because that line uses build isolation (no --no-build-isolation flag),
    # PyPI only ships a source tarball for flash-attn, and the isolated build env
    # doesn't have torch → ModuleNotFoundError. The prebuilt GitHub-released wheel
    # works around it. Once installed, setup.sh's pip line is a no-op.
    if [[ "$TRELLIS_FLAGS" == *"--new-env"* ]]; then
        # Need the env created first. Run just --new-env, then pre-install flash-attn, then everything else.
        . ./setup.sh --new-env
        conda activate trellis2
        log "pre-installing flash-attn $FLASH_ATTN_VER prebuilt wheel (avoids PyPI source-build trap)"
        pip install "$FLASH_ATTN_WHL"
        # Now run the rest without --new-env
        . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    else
        # Env already exists with correct torch — same flash-attn workaround applies.
        if ! python -c "import flash_attn, sys; sys.exit(0 if flash_attn.__version__ == '$FLASH_ATTN_VER' else 1)" 2>/dev/null; then
            log "pre-installing flash-attn $FLASH_ATTN_VER prebuilt wheel"
            pip install "$FLASH_ATTN_WHL" --upgrade
        fi
        . ./setup.sh $TRELLIS_FLAGS
    fi

    ok "trellis2 setup.sh finished"
fi

# ---- 6. verification — fail loud if any extension didn't actually install --

log "Step 6/6: verify every extension imports AND a kernel actually runs on GPU"

VERIFY_FAILED=0
# Correct installed module names: nvdiffrec ships as nvdiffrec_render; FlexGEMM as flex_gemm
for mod in torch flash_attn nvdiffrast nvdiffrec_render cumesh flex_gemm o_voxel; do
    if conda run -n trellis2 python -c "import $mod" 2>/dev/null; then
        ok "$mod imports"
    else
        warn "$mod FAILED to import"
        VERIFY_FAILED=1
    fi
done

# Actually run a kernel on the GPU. Without this we'd accept a torch install
# that's the wrong arch (e.g. sm_120 missing) since import alone succeeds.
if conda run -n trellis2 python -c "
import torch
assert torch.cuda.is_available(), 'cuda not available'
cap = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
sm = f'sm_{cap[0]}{cap[1]}'
assert sm in arches, f'torch wheel does not include {sm}; has {arches}'
a = torch.randn(1024, 1024, device='cuda')
b = (a @ a).sum().item()
print(f'  device cap {sm}, arches {arches}, matmul sum {b:.2f}')
" 2>&1; then
    ok "GPU matmul executes on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "GPU matmul FAILED — torch wheel may not support this device's architecture"
    VERIFY_FAILED=1
fi

if [ "$VERIFY_FAILED" -eq 1 ]; then
    fail "one or more TRELLIS.2 components are broken. See /tmp/trellis2_setup.log."
fi

log "Done."
echo ""
echo "Next steps:"
echo "  1. conda activate trellis2"
echo "  2. cd ~/projects/image_to_splat"
echo "  3. python image_to_splat_dataset.py --help"
