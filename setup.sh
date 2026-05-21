#!/usr/bin/env bash
# Image2Splat top-level installer. Orchestrates:
#   1. TRELLIS.2 clone + conda env (via scripts/setup_env.sh)
#   2. Pixal3D clone (TencentARC's productized fork of TRELLIS.2)
#   3. AuraSR pip install in a sibling conda env for the upscaler
#   4. Interactive prompt for IMAGE2SPLAT_HOTFOLDER env var
#
# Re-run safe: each step skips work already done.
# Total time: ~60-90 min (most of it conda env builds + model downloads).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECTS_DIR="$HOME/projects"
PIXAL3D_DIR="$PROJECTS_DIR/Pixal3D"

color() { printf "\033[1;%sm%s\033[0m\n" "$1" "$2"; }
log()   { color 36 "==> $*"; }
ok()    { color 32 "  ✓ $*"; }
warn()  { color 33 "  ! $*"; }
fail()  { color 31 "  ✗ $*"; exit 1; }

log "Image2Splat installer"
log "  repo: $REPO_DIR"
log "  installs to: $PROJECTS_DIR"
echo

# -----------------------------------------------------------------------------
log "Step 1/4: TRELLIS.2 + conda env (delegated to scripts/setup_env.sh)"
# -----------------------------------------------------------------------------
bash "$REPO_DIR/scripts/setup_env.sh"
ok "TRELLIS.2 env ready"

# -----------------------------------------------------------------------------
log "Step 2/4: clone Pixal3D"
# -----------------------------------------------------------------------------
if [ -d "$PIXAL3D_DIR/.git" ]; then
    ok "Pixal3D already cloned at $PIXAL3D_DIR"
else
    git clone https://huggingface.co/TencentARC/Pixal3D "$PIXAL3D_DIR" || \
        fail "Pixal3D clone failed — check Hugging Face access"
    ok "Pixal3D cloned"
fi

# -----------------------------------------------------------------------------
log "Step 3/4: AuraSR upscaler env (anysplat conda env, side-by-side with trellis2)"
# -----------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
if conda env list | grep -q "^anysplat "; then
    ok "anysplat env already exists"
else
    log "  creating anysplat conda env (Python 3.10 + torch 2.7 + cuda 12.8)"
    conda create -y -n anysplat python=3.10
    conda activate anysplat
    pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
    pip install aura-sr realesrgan basicsr spandrel
    ok "anysplat env ready"
fi

# -----------------------------------------------------------------------------
log "Step 4/4: configure IMAGE2SPLAT_HOTFOLDER"
# -----------------------------------------------------------------------------
SHELL_RC="$HOME/.bashrc"
[ -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.zshrc"

if grep -q "IMAGE2SPLAT_HOTFOLDER" "$SHELL_RC" 2>/dev/null; then
    ok "IMAGE2SPLAT_HOTFOLDER already set in $SHELL_RC"
else
    DEFAULT_HOTFOLDER="$HOME/image2splat"
    # If running in WSL, suggest a Desktop folder
    if grep -qi microsoft /proc/version; then
        WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
        if [ -n "$WIN_USER" ]; then
            DEFAULT_HOTFOLDER="/mnt/c/Users/$WIN_USER/Desktop/image2splat"
        fi
    fi
    echo
    read -p "Hot-folder location [default: $DEFAULT_HOTFOLDER]: " HOTFOLDER
    HOTFOLDER="${HOTFOLDER:-$DEFAULT_HOTFOLDER}"
    echo "" >> "$SHELL_RC"
    echo "# Image2Splat hot-folder" >> "$SHELL_RC"
    echo "export IMAGE2SPLAT_HOTFOLDER='$HOTFOLDER'" >> "$SHELL_RC"
    mkdir -p "$HOTFOLDER"
    ok "IMAGE2SPLAT_HOTFOLDER added to $SHELL_RC"
    ok "hotfolder created: $HOTFOLDER"
fi

echo
log "Install complete."
echo
echo "Next steps:"
echo "  1. Reload your shell:  source $SHELL_RC"
echo "  2. (Windows-WSL) copy BAT files to Desktop:"
echo "       cp '$REPO_DIR/bat/Start Splat Daemon.bat' /mnt/c/Users/<you>/Desktop/"
echo "       cp '$REPO_DIR/bat/Start Upscale.bat' /mnt/c/Users/<you>/Desktop/UPSCALE/"
echo "  3. Drop images into your hotfolder and run the daemon"
echo
echo "  See README.md for full usage."
