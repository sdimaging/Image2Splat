# Image2Splat

**Single-image to 200-view splat-training dataset**, with a per-asset aesthetic dialer.

Drop a photo of an object into a folder. Get back a 200-view orbit-rendered COLMAP + Nerfstudio dataset, ready to train into a 3D Gaussian splat in [PostShot](https://www.jawset.com/), [LichtFeld](https://lichtfeld.io), [gsplat](https://github.com/nerfstudio-project/gsplat), or any other 3DGS trainer.

Built on [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (via [TencentARC's Pixal3D](https://huggingface.co/TencentARC/Pixal3D) productized fork). Adds:

- **A 5-tier sampler system** with empirically-validated configurations for different asset types (organic/stone/weathered → hard-surface/mechanical → max sculpt detail)
- **A probe mode** that runs all 5 tiers on a single asset at view 129, lets you visually compare and dial in the right aesthetic *before* committing to a full 200-view render
- **A seed-quantifier** for the probe so you can also see seed-to-seed variation per tier
- **An AutoCrop preprocessor** that tight-crops inputs to a 40px-bordered subject bbox — eliminates wasted negative space so the subject occupies more of Pixal3D's 1024-pixel image conditioning
- **An AuraSR-based generative upscaler** for preprocessing soft / low-res / compressed source images before they go through the splat pipeline
- **A hot-folder daemon** that watches a folder for incoming images and processes them with one or many tier configurations
- **Drop-in Windows BAT files** for daemon + autocrop + upscaler, so the whole workflow is double-click-from-desktop

## Why this exists

Most image-to-3D tools (TRELLIS.2 demo, LiTo, AnySplat, Rodin Gen-2.5) give you one output per image — a single SLat-decoder-baked aesthetic. **Image2Splat treats Pixal3D's sampler params as creative knobs** rather than fixed defaults, surfacing them as named tiers that map to genuine per-asset-type aesthetics.

The 5 tiers were locked in through ~15 hours of sweeps on Gargoyle / Knight / Column / and other reference assets:

| Tier | Steps | Shape guidance | Tex guidance | Best for |
|---|---|---|---|---|
| 1 — **Default** | 12 | 7.5 | 1.1 | Safe production baseline |
| 2 — **Subtle** | 13 | 7.6 | 1.2 | Mild bump, conservative |
| 3 — **Balanced** | 14 | 7.7 | 1.25 | Mathematical midpoint |
| 4 — **Refined** | 15 | 7.6 | 1.2 | Hard-surface + organic carved-detail |
| 5 — **Sculpted** | 15 | 8.0 | 1.5 | Max push, hard-surface, mechanical |

Pixal3D's published defaults are `12/7.5/1.0` (sparse_structure / shape_slat / tex_slat). Empirically the tiny tex bump in Tier 1 produces visibly cleaner texture for our use cases.

## What you get out

Per image dropped in `inbox/`:

```
datasets/<image_slug>/USER_Alt2_200v_3000px/
  images/                  ← 200 PNG renders @ 3000×3000, orbital camera trajectory
    000.png  ...  199.png
  transforms.json          ← Nerfstudio-format camera intrinsics + extrinsics
  cameras.txt              ← COLMAP-format extrinsics (binary fmt also written)
  images.txt
  points3D.txt
```

This is **NOT a finished splat** — it's the training dataset. You then feed it to your splat trainer of choice:
- PostShot (paid, Windows, fast, includes COLMAP-bypass)
- LichtFeld (paid, Windows/Mac, similar)
- gsplat / nerfstudio / 3DGS (free, CLI, more setup)

A note on PostShot v1.1+: it requires a track-count alignment that the daemon's output doesn't auto-satisfy. Run `scripts/fix_colmap_postshot11.py` on your output dataset before importing.

## Quickstart

### Requirements

- **OS:** WSL2 on Windows 10/11, or Linux. (BAT files assume WSL on Windows.)
- **GPU:** NVIDIA RTX with ≥24 GB VRAM. Validated on RTX 5090 (sm_120 Blackwell). Should work on 3090 / 4090 / A6000 / H100. Compute time scales inversely with GPU speed.
- **CUDA:** 12.4+ (for sm_120 you need 12.8). Auto-managed by the conda env.
- **Disk:** ~50 GB for env + models + ~10 GB per processed asset (200 PNG renders × 3000×3000)
- **Time per asset:** ~5-7 min mesh+render on 5090, ~10-15 min total once you include polish/colmap export.

### Install

```bash
# 1. Clone this repo
git clone https://github.com/sdimaging/Image2Splat.git
cd Image2Splat

# 2. Run the env setup (clones Pixal3D + TRELLIS.2, creates conda env)
bash scripts/setup_env.sh

# 3. (Windows-WSL) Copy the BAT files to your hot-folder on Desktop
#    Recommended nested layout — everything in one folder for the splat workflow:
#
#    Desktop\image2splat\
#      inbox\           ← drop source images here
#      processing\      ← daemon work-in-flight
#      completed\       ← successfully processed sources
#      failed\          ← errors + .error.log
#      datasets\        ← COLMAP/Nerfstudio output datasets
#      UPSCALE\         ← (optional) AuraSR preprocessor folder
#        Input\
#        Output\
#        Start Upscale.bat
#      Start Splat Daemon.bat
#      Autocrop.bat
#
cp bat/"Start Splat Daemon.bat" "$IMAGE2SPLAT_HOTFOLDER/"
cp bat/"Autocrop.bat"           "$IMAGE2SPLAT_HOTFOLDER/"
mkdir -p "$IMAGE2SPLAT_HOTFOLDER/UPSCALE"
cp bat/"Start Upscale.bat"      "$IMAGE2SPLAT_HOTFOLDER/UPSCALE/"
```

**Nesting UPSCALE inside the hot-folder is the recommended layout** — keeps the entire splat workflow toolset (autocrop → upscale → daemon) in one place. The UPSCALE BAT auto-detects its own location, so it can live anywhere on disk. If you want it as a standalone day-to-day image upscaler not tied to the splat workflow, you can put it wherever you like — `Desktop\UPSCALE\` works fine too.

### Configure

Set the hot-folder location (where dropped images get processed):

```bash
# In your ~/.bashrc or ~/.zshrc
export IMAGE2SPLAT_HOTFOLDER='/mnt/c/Users/<you>/Desktop/image2splat'
```

The daemon will auto-create `inbox/`, `processing/`, `completed/`, `failed/`, and `datasets/` subfolders inside this directory on first run.

### Run the daemon

Either:
- Double-click `Start Splat Daemon.bat` (Windows) — opens a terminal, prompts you for seed / HDRI / tier / etc.
- Or: `bash scripts/run_hotfolder.sh` in WSL directly

You'll get this prompt sequence:
```
Seed [default 222]:
HDRI: <list of EXRs>
Force background removal even on already-masked PNGs? [y/N]:
Sampler tier:
  1) Default    steps=12 shape=7.5 tex=1.1
  2) Subtle     steps=13 shape=7.6 tex=1.2
  3) Balanced   steps=14 shape=7.7 tex=1.25
  4) Refined    steps=15 shape=7.6 tex=1.2
  5) Sculpted   steps=15 shape=8.0 tex=1.5
  6) Probe      runs ALL 5 tiers at view 129 (multi-tier comparison)
Select [1-6, default 1]:
```

Then drop images into `inbox/`. Daemon picks them up, processes, moves source to `completed/`, writes dataset to `datasets/`.

### Probe mode (option 6)

Don't know which tier suits an asset? Drop it in, pick **6 — Probe**:

```
Probe seeds (1 = just 222, N = 222 + N-1 randoms):
Select [1-8, default 1]:
```

Daemon renders **view 129 only** at **5 tiers × N seeds** = 5N comparison frames at `datasets/<slug>/probe/`, plus `probe_meta.json` for reproducibility.

Open the folder in Photoshop / Bridge as layers, flip between tiers + seeds, pick the winner. Then re-drop the same image at that specific tier for the full 200v render.

### Optional: AutoCrop before splat pipeline (recommended for catalog-style sources)

If your inputs have white or black backgrounds with negative space around the subject (typical for product photography, museum catalog images, AI-generated stock), AutoCrop eliminates wasted pixels so the subject occupies more of Pixal3D's 1024-pixel image conditioning.

1. Drop images into the daemon's `inbox/` folder
2. Double-click `Autocrop.bat`
3. Originals are auto-backed up to `inbox/.original_backups/` and replaced with tight-cropped versions (40px margin around subject)
4. Then run the daemon as normal

How it detects the subject: pixel-statistic bbox detection — any pixel that's NOT near-white (RGB > 240), NOT near-black (RGB < 15), and NOT transparent (alpha < 10) counts as subject. Expands the bbox by 40px on all sides for clean margins.

**Subject pixel density gain**: a 2000×2000 input where the subject is the center 800×800 becomes a ~880×880 crop where the subject fills nearly the entire frame. Pixal3D's downsample-to-1024 then sees ~1000 pixels of subject instead of ~400. ~5-6× effective density boost with zero hallucination.

### Optional: AuraSR upscale before splat pipeline

For soft / low-res / compressed source images, the AuraSR upscaler adds plausible micro-detail that SURVIVES Pixal3D's internal 1024-pixel image conditioning downsample:

1. Drop source images in `Desktop\UPSCALE\Input\`
2. Double-click `Start Upscale.bat`
3. Get 4× upscaled PNGs in `Desktop\UPSCALE\Output\`
4. Use those as the new inputs to the splat daemon

GAN-based generative upscale — adds detail it thinks should be there. Best for inputs that look soft. Skip for already-crisp 2K+ images.

**Stack with AutoCrop for max benefit**: AutoCrop first (eliminate negative space), then AuraSR (4× detail on the cropped subject), then daemon. Each stage multiplies effective subject density at Pixal3D's conditioning stage.

## Workflow patterns

### A — Single tier production
```
drop image → pick tier 1-5 → 200v at that tier → train splat
```

### B — Probe first, then production
```
drop image → tier 6 probe → 5 comparison frames per seed
→ pick best (tier, seed) from probe → re-drop at that tier → 200v
→ train splat
```

### C — Multi-tier production
```
drop image → pick tier 1 → 200v at tier 1 → move source back to inbox
→ drop again, pick tier 4 → 200v at tier 4 → ...
→ cherry-pick views across tier outputs in Photoshop before splat training
→ get a custom "best of N tiers" splat that no single sampler config could produce
```

### D — Full preprocessing chain (max subject density)
```
drop catalog-style images in inbox → Autocrop.bat → tight-cropped versions in place
→ (move to UPSCALE\Input) → Start Upscale.bat → 4× AuraSR'd versions in UPSCALE\Output
→ (move back to inbox) → Start Splat Daemon.bat → pick tier → 200v at chosen tier
→ train splat
```
Each preprocessing stage multiplies effective subject pixel density at the conditioning stage:
1. **AutoCrop**: ~5-6× density gain by eliminating negative space
2. **AuraSR**: 4× supersample (artifacts get filtered out by Pixal3D's 1024 downsample)
3. **Pixal3D**: subject fills ~95%+ of the 1024×1024 conditioning frame instead of ~20-40%

### E — Generative-upscale + probe
```
soft / compressed source → AuraSR upscale → SR'd version into splat daemon
→ probe mode for tier selection → 200v at chosen tier → splat
```

## Architecture

```
your image
   │
   ▼
[ optional: AutoCrop  → eliminate negative space ]
   │
   ▼
[ optional: AuraSR upscale  → 4× generative detail ]
   │
   ▼
hotfolder_daemon.py
   │  • image preprocess (BiRefNet bg removal if no alpha)
   │  • MoGe camera estimation
   │  • Pixal3D run (sparse → shape → texture SLat sampling)
   │  • mesh extraction (~6-21 M faces, QEM-decimated to ≤16.78M)
   │  • multiview rendering (200v Fibonacci orbit, 3000px, HDRI lighting)
   │  • polish pass (sharpen + film grain)
   │  • COLMAP + Nerfstudio dataset export
   ▼
datasets/<slug>/USER_Alt2_200v_3000px/
   │
   ▼
[ your splat trainer: PostShot / LichtFeld / gsplat ]
   │
   ▼
final 3D Gaussian splat
```

## Repository layout

```
src/                     ← Python source for the splat pipeline
  backbone.py              Pixal3D / TRELLIS.2 backbone wrapper
  cameras.py               Fibonacci-sphere camera trajectory + intrinsics
  render.py                Multi-view PBR rendering via nvdiffrast
  polish.py                Per-view sharpen + grain post-process
  hdri_io.py               EXR HDRI envmap loader
  mesh_export.py           PLY + GLB export
  export_colmap.py         COLMAP dataset writer
  export_nerfstudio.py     Nerfstudio transforms.json writer

scripts/
  hotfolder_daemon.py      The main daemon
  run_hotfolder.sh         WSL launcher for the daemon
  setup_env.sh             Install Pixal3D + TRELLIS.2 + conda env
  autocrop.py              Pixel-stat bbox autocropper for tight-cropping
  run_autocrop.sh          WSL wrapper for autocrop
  probe_tier_validation.py Reproducer for the tier-locking sweep
  fix_colmap_postshot11.py PostShot v1.1+ COLMAP track-count patcher

upscale/
  upres.py                 AuraSR folder-to-folder upscaler
  run_upscale.sh           WSL wrapper

bat/                       Windows BAT files (copy to Desktop)
  Start Splat Daemon.bat
  Autocrop.bat
  Start Upscale.bat

docs/                      Design notes + eval sessions
```

## Tunable params reference

| Param | Default | Notes |
|---|---|---|
| seed | 222 | Production-locked. Different seeds = different outputs at same tier. |
| HDRI | USER_Alt2.exr | Studio HDRI used for shading. Swap for different lighting. |
| HDRI exposure | 2.5× | Linear multiplier on radiance. |
| Pipeline | 1536_cascade | 1536 = HR shape SLat at 1536³ voxel grid. 2048 is OOD and may fail. |
| Max tokens | 131,072 | 128k token budget for sparse-voxel structure. |
| Max faces | 16,000,000 | Cap before QEM decimation. nvdiffrast hard cap = 2^24 = 16.78M. |
| Resolution | 3000 px | Per-view render resolution. PostShot prefers 2-4K. |
| Num views | 200 | Fibonacci-sphere orbit. 200 is the validated count for object splats. |
| FOV | 40° | Camera horizontal field of view. |
| Polish sharpen | 40% | Subtle unsharp. Disabled via 0. |
| Polish grain | σ=3.0 | Gaussian noise post-render. Disabled via 0. |

## What this DOESN'T do

- ❌ **Train the splat** — output is the dataset, you bring your own trainer (PostShot, LichtFeld, gsplat, etc.)
- ❌ **Multi-image input** — single image to single asset. Use TRELLIS.2 directly for multi-image conditioning.
- ❌ **Text-to-3D** — image-to-3D only.
- ❌ **Editing existing splats** — generation only, no remixing.

## Known limitations + roadmap

- **GPU-locked** — needs CUDA. Apple Silicon / CPU paths are untested.
- **Single-asset orientation** — assumes the input image's principal axis maps to the asset's vertical. Auto-aligned via MoGe; for unusual poses you may need to pre-rotate the input.
- **Tier 6 probe currently single-view (129)** — could extend to multi-view probe later for trickier assets.

## Credits

- [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) — foundational image-to-3D model architecture
- [TencentARC Pixal3D](https://huggingface.co/TencentARC/Pixal3D) — productized TRELLIS.2 fork with pixel-aligned back-projection
- [valeoai/MoGe](https://github.com/valeoai/MoGe-2) — monocular geometry estimation
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — background removal
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) — differentiable rendering
- [fal/AuraSR](https://huggingface.co/fal/AuraSR-v2) — generative super-resolution
- [CuMesh](https://github.com/JeffreyXiang/CuMesh) — GPU mesh ops + QEM decimation

## License

MIT — see [LICENSE](LICENSE).
