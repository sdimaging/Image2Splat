# Install

Turnkey install for Windows + WSL2 (the tested path). Linux-native works too —
skip the Windows/BAT notes.

## Prerequisites
- **NVIDIA GPU**, 24 GB+ VRAM recommended (tested on RTX 5090, 32 GB).
- **WSL2 with Ubuntu** (Windows) or a Linux box.
- **Miniconda/Anaconda** installed in WSL (`~/miniconda3` or `~/anaconda3`).
- A **Hugging Face account** with access to `TencentARC/Pixal3D`.
- ~60 GB free disk (models + conda envs).

## One-shot install

```bash
# In WSL (clone wherever you like — the installer adapts to the location):
git clone https://github.com/sdimaging/Image2Splat ~/projects/Image2Splat
cd ~/projects/Image2Splat
./setup.sh
```

`setup.sh` is re-run safe and does everything:
1. Clones **TRELLIS.2** + builds the `trellis2` conda env.
2. Clones **Pixal3D** (TencentARC).
3. Builds the `anysplat` conda env for the AuraSR upscaler.
4. Asks **where you want your hot-folder** and saves `IMAGE2SPLAT_HOTFOLDER`
   to your shell profile.
5. **Creates the hot-folder tree** (`inbox/ processing/ completed/ failed/
   datasets/ UPSCALE/`) and **installs the launcher BATs into it**, with this
   repo's path baked in — so they work no matter where you cloned.

> On WSL, pick a **Windows path** for the hot-folder (e.g. a Desktop folder) so
> the `.bat` launchers are double-clickable from Windows.

## Run

```bash
source ~/.bashrc          # load IMAGE2SPLAT_HOTFOLDER (first time only)
```

Then open your hot-folder and **double-click `Start Splat Daemon.bat`**.
- Drop images into `inbox\`.
- Pick a run type at the prompt: `1-5` (production tier), `6` (probe), `B` (batch).
- The run self-heals — auto-restarts on OOM/GPU resets, quarantines a bad asset
  after 3 strikes, and resumes after a reboot. Datasets land in `datasets\`.

Linux-native (no BAT): run the launcher directly —
```bash
bash ~/projects/Image2Splat/scripts/run_hotfolder_watchdog.sh --ask
```

## Re-running / moving the hot-folder
Re-run `./setup.sh` any time — it re-installs the BATs against the current repo
path. To move the hot-folder, edit `IMAGE2SPLAT_HOTFOLDER` in your shell profile
(or just re-run `setup.sh` after removing that line) and re-run the installer.

See [README.md](README.md) for full usage, workflow patterns, and tuning.
