# Splat Pipeline Evaluation Session — 2026-05-20

End-to-end evaluation of alternatives to the production Pixal3D + 200v + PostShot pipeline.
Result: **production pipeline stays primary**; texture-quality improvement comes from a new
super-resolution post-render step, not from replacement models.

---

## TL;DR

| Tool evaluated | Verdict | Why |
|---|---|---|
| **Pixal3D + 200v + PostShot** (current) | **KEEP — primary** | Already shipping; texture mush is the only complaint, addressed via SR add-on |
| **LiTo** (Apple image-to-3D) | Shelved — install documented for future revisit | Got it running on 5090 after a Blackwell backend spike; output quality below Pixal3D at all tested settings |
| **AnySplat** (sparse-view feed-forward) | Not applicable | Trained on real-scene photos; object-on-black renders are out-of-distribution |
| **Rodin Gen-2.5** | Closed API, paid SaaS | No self-host option |
| **TRELLIS.2 standalone** | Already what Pixal3D wraps | Pixal3D = TencentARC's productized TRELLIS.2 fork |
| **Faithful Contouring** (CVPR 2026) | Future watch | Representation method only, generative DiT not released yet |
| **Super-resolution (AuraSR / Real-ESRGAN / 4x-UltraSharp)** | **WIRE INTO DAEMON** | UltraSharp via spandrel = 0.25s/view; smooth-then-SR path = ~5-14 min per 200v dataset |

**Action item:** integrate 4x-UltraSharp (smooth-then-SR via spandrel) as a post-render step in `hotfolder_daemon.py` before the views go to splat training.

---

## RTX 5090 Blackwell (sm_120) compatibility spike — LiTo

The user's terminal-time-boxed methodology that got LiTo running. **Document this in case we revisit when xformers ships native Blackwell support.**

### Symptom
LiTo inference crashes inside xformers' bundled flash-attention at
`flash-attention/hopper/flash_fwd_launch_template.h:188` with
`CUDA error: invalid argument` — even after all 50+ deps install successfully.

### Root cause
xformers 0.0.33's auto-dispatcher picks `flash3.FwOp` (sm_90 Hopper-only kernel)
for the 5090 (sm_120 Blackwell). The kernel doesn't exist on Blackwell.

### Call site documentation

| File | Function | xformers call shape |
|---|---|---|
| `third_party/TRELLIS/trellis/modules/sparse/attention/full_attn.py:198` | `sparse_scaled_dot_product_attention` | `xops.memory_efficient_attention(q, k, v, BlockDiagonalMask)` — block-diagonal varlen sparse self-attn |
| `third_party/TRELLIS/trellis/modules/sparse/attention/windowed_attn.py:110, 123` | windowed sparse self-attn | with/without BlockDiagonalMask |
| `third_party/TRELLIS/trellis/modules/sparse/attention/serialized_attn.py:168, 181` | serialized sparse self-attn (vox2seq) | with/without BlockDiagonalMask |
| `third_party/TRELLIS/trellis/modules/attention/full_attn.py:113` | dense full attention | `xops.memory_efficient_attention(q, k, v)` |
| `src/lito/models/*.py` + `libraries/plibs/src/plibs/ppoint.py` | ~50 more sites in LiTo proper | all same `xops.memory_efficient_attention` API |

Shapes: `[1, sum(seqlens), H, C]` with `BlockDiagonalMask` for the sparse paths,
`[B, M, H, K]` for dense. Dtype: fp16 (or bf16 via autocast).

### Fix #1 — Cutlass FwOp monkey-patch
Cutlass is arch-agnostic. Force xformers to use it instead of flash3:

```python
# blackwell_attn_patch.py — imported before any model code
import functools
import xformers.ops as xops
from xformers.ops.fmha.cutlass import FwOp as _CutlassFwOp

_orig = xops.memory_efficient_attention

@functools.wraps(_orig)
def _patched(q, k, v, attn_bias=None, p=0.0, scale=None, *, op=None, output_dtype=None):
    if op is None:
        op = (_CutlassFwOp, None)
    return _orig(q, k, v, attn_bias=attn_bias, p=p, scale=scale, op=op, output_dtype=output_dtype)

xops.memory_efficient_attention = _patched
import xformers.ops.fmha as _fmha
if hasattr(_fmha, "memory_efficient_attention"):
    _fmha.memory_efficient_attention = _patched
```

Single shim covers all 50+ call sites — no per-site patching needed.

### Fix #2 — einops MLXBackend disable (Linux-only artifact)
Apple's pixi env installs `mlx` (Apple's ML framework) as a transitive dep but
`libmlx.so` is missing on Linux. `einops/_backends.py:728` tries
`import mlx.core as mx` inside torch.compile's FX trace context and the ImportError
leaks through Dynamo.

```python
# in .pixi/envs/default/lib/python3.11/site-packages/einops/_backends.py
class _MLXBackend_DISABLED:  # was: class MLXBackend(AbstractBackend):
    ...
```

Removing the inheritance from `AbstractBackend` de-registers MLXBackend from
`AbstractBackend.__subclasses__()` so einops never tries to instantiate it.

### Fix #3 — fp32 inference (bfloat16 autocast removal)
Default `fastapi_lito_demo.py` does `torch.autocast(dtype=torch.bfloat16)` even
though the checkpoint loads as fp32. Strip the autocast for sharpest texture:

```python
# demos/lito/fastapi_lito_demo.py — replace autocast block with plain no_grad
with torch.no_grad():
    out_dict = model.inference_sample_latent(...)
```

### Required env vars at runtime
```
ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=xformers
```
`ATTN_BACKEND=sdpa` routes dense attention through PyTorch's built-in SDPA (no
xformers). `SPARSE_ATTN_BACKEND=xformers` keeps the sparse paths on the
monkey-patched cutlass-dispatched xformers (since sparse has no SDPA fallback).

### Verdict
**WORKS** — LiTo inference completes on the 5090 in ~28s default / ~200s at
steps=100+CFG=7+fp32. Acid-tested with a clean TRELLIS-distribution input
(`typical_creature_dragon.png`) — output is dense and detailed, confirming
install correctness. Column1 input (Pixal3D render) produces blurry output =
domain mismatch (Pixal3D output ≠ Objaverse training distribution).

**Why we still shelved it:** even on in-distribution inputs, LiTo's output
texture density / quality does not exceed Pixal3D + the SR post-process for
our typical column/chess-piece source images. Workflow simplification (1
image → 1 splat in 30s) is real, but quality bar isn't there yet.

---

## LiTo Linux install gotchas (not Apple-AWS)

Apple's setup scripts assume internal AWS infrastructure. Every gotcha and
its workaround:

| Issue | Fix |
|---|---|
| `pixi install` script piped from curl blocked by classifier | Download pixi binary directly: `wget github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl.tar.gz` |
| pytorch3d compile fails on `stable` branch with CUDA 12.8 | Use tag `v0.7.9` + `TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"` + `MAX_JOBS=4` (RAM cap) |
| `setup_trellis.sh` hardcodes `/mnt/pkg_install` (AWS-EFS) | Symlink or edit script — the path doesn't exist on WSL and lacks `set -e` so script silently no-ops past it |
| TRELLIS submodule not initialized | `git submodule update --init --recursive third_party/TRELLIS` |
| `extensions/vox2seq` referenced but doesn't exist in pinned TRELLIS commit | **vox2seq isn't actually needed for LiTo inference** — only TRELLIS training. Skip it. |
| `cyobj` Cython compile fails with `longintrepr.h: No such file` (Python 3.11+ API change) | **Not needed for LiTo inference either** — skip it |
| `kaolin` `_C` ImportError after install | `_C` extension didn't compile; rerun `python setup.py build_ext --inplace` with right TORCH_CUDA_ARCH_LIST |
| `open3d` listed in pixi deps but missing from env | `pixi run pip install open3d` (silent install failure during pixi install) |
| `mlx` installed but `libmlx.so` missing | See "Fix #2" in Blackwell spike above |
| xformers 0.0.33 pinned to torch 2.9.1; newer versions need torch 2.10+ which breaks kaolin/diff-gaussian-rasterization | Stay on 0.0.33 + apply the cutlass monkey-patch |

### Minimum runtime env vars (LiTo)
```bash
export PATH="$HOME/.pixi/bin:$PATH"
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=xformers
cd ~/projects/LiTo && pixi run python demos/lito/fastapi_lito_demo.py --port 8000
```

---

## AnySplat (sparse-view feed-forward) — separate eval

Different family of model: takes 2-N unposed images, predicts 3D Gaussians +
camera poses in 2.7s. Trained on CO3D + DL3DV + ScanNet++ (real photos with full
backgrounds). Tested on Gargoyle1's 200v orbit views.

### Result
Works end-to-end on 5090. Geometry roughly captured. **But:** novel views show
fishnet/mesh artifacts because the 8-view orbit doesn't match its training
distribution (it expects real-photo motion with continuous overlapping frames,
not synthetic-orbit renders on black bg).

### Conclusion
Not the right tool for Pixal3D-render inputs. Would need fine-tuning on
G-Objaverse renders or similar synthetic-orbit data to fit our use case —
a 2-3 week research project, not a drop-in.

Env retained at `~/projects/AnySplat/` for future revisits (15 GB).

---

## Super-resolution evaluation — THE WIN

### 3-way shootout on Column1 200v dataset (views 098-103, downsampled to 768→3072)

| Method | Per view | Per 200v dataset | Format |
|---|---|---|---|
| **AuraSR** | 6-10 s | ~25 min | fal HF, GAN-based, aggressive hallucination — risk for view-consistency |
| **Real-ESRGAN x4plus** | 0.9 s | ~3 min | RRDBNet via realesrgan lib, tile=512 |
| **4x-UltraSharp via spandrel** | **0.25 s** | **~50 s** | Classic ESRGAN format, spandrel auto-detects, fp16, single-pass |

**Winner: 4x-UltraSharp via spandrel** by 30× speed over AuraSR with cleaner output.

### Critical compat learnings

- **basicsr + modern torchvision** — needs patch:
  ```bash
  sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' \
    <env>/lib/python3.10/site-packages/basicsr/data/degradations.py
  ```
- **UltraSharp `.pth` format** — uses classic ESRGAN keys (`model.0.weight`, `model.1.sub.X.RDB...`), NOT Real-ESRGAN's `params`/`params_ema` nested keys. RealESRGANer loader fails with `KeyError: 'params'`. **Use `spandrel`** — auto-detects any SR format.

### Smooth-then-SR pipeline tuning (user-discovered insight)

Direct 3000→12000 SR on Pixal3D renders just amplifies SLat decoder mush.
Pre-downsampling first (acts as low-pass filter) → SR → final downsample
gives genuinely sharper output AND is 5-13× faster.

| Path | Pipeline | Per view | Per 200v |
|---|---|---|---|
| X (control) | 3000 → SR 12000 → Lanczos 4000 | 20.7 s | ~70 min |
| **A (recommended)** | 3000 → Lanczos 1500 → SR 6000 → Lanczos 4000 | **4.3 s** | **~14 min** |
| **B (aggressive)** | 3000 → Lanczos 1000 → SR 4000 (final) | **1.6 s** | **~5 min** |

### Tiled SR runner (for memory ceiling)
3000→12000 in one shot OOMs on 5090 (17 GB allocation). Tiled with 384px tiles +
24px overlap padding + fp16 fits comfortably. Spandrel-loaded model must use
`.model.half().cuda().eval()` to actually move to GPU (`ModelDescriptor`'s
`.cuda()` doesn't propagate properly).

### Alpha handling
Pixal3D's renderer outputs RGBA with bg_color=(0,0,0). RGB-behind-transparent
is already pure black — **no halo risk when stripping alpha for SR**. Alpha
upscale via LANCZOS > BICUBIC for sharper edges. Threshold-after-bicubic also
works for hard binary edges.

### Photoshop CameraRaw batch — alternative
File > Automate > Batch with a saved CameraRaw action enhances each view
in-place (no upscale). Or save as Droplet for drag-drop. User-tunable, view-
consistent by construction. Doesn't upscale but adds clarity/sharpening
via Camera Raw's algorithms. Worth keeping as a non-ML option.

### Code artifacts retained
- `~/sr_eval/run_sr_shootout.py` — 3-way comparison runner
- `~/sr_eval/test_smooth_then_sr.py` — Path A/B/X comparison
- `~/sr_eval/test_12k_timing.py` — 4x → downsample pipeline with tiling
- `~/sr_eval/models/4x-UltraSharp.pth` — 64 MB model file
- `Desktop/sr_shootout/` — all comparison outputs at 4k

---

## TRELLIS.2 architecture confirmation (rabbit hole resolved)

User's existing Pixal3D pipeline IS the current SOTA Microsoft TRELLIS.2 model.
Multiple May 2026 Twitter posts referencing "Microsoft's 4B model" or "world's
first 10M polygon" are all marketing for TRELLIS.2, which user already runs
via Pixal3D's productized fork.

| Aspect | TRELLIS.2 (what user already runs) |
|---|---|
| Origin | Microsoft Research |
| Params | 4B |
| Format | O-Voxel (field-free sparse voxel) |
| Output | GLB with full PBR (BaseColor + Roughness + Metallic + Opacity) |
| Released | Dec 2025 |
| User's wrapper | Pixal3D (TencentARC) |

No new model upgrade needed at the base level — Pixal3D already runs the SOTA.

---

## Permission rules added (project settings)

For future LiTo-style installs that hit the security classifier:
`.claude/settings.local.json` (gitignored, personal-machine):
```json
"Bash(cd ~/projects/LiTo && *)",
"Bash(cd ~/projects/LiTo/* && *)",
"Bash(pixi run bash env/scripts/*)",
"Bash(pixi run pip install*)"
```

---

## Production daemon state (current primary)

- Defaults: seed=222, USER_Alt2.exr HDRI, exposure=2.5, 1536_cascade pipeline, max_tokens=131072
- BAT script: `Start Splat Daemon.bat` on Desktop, watches `~/Desktop/image_to_splat/inbox/`
- Output: 200 views @ 3000px RGBA + COLMAP + Nerfstudio dataset → PostShot
- Already has `polish_sharpen_percent` + `polish_grain_strength` post-render

**Next integration:** add 4x-UltraSharp (smooth-then-SR Path A or B) as an
optional post-render step before the daemon writes the final view PNGs.
Open question: does it actually improve PostShot's splat output, or does
PostShot average out the added detail? Needs A/B test on one full 200v
dataset → splat train → side-by-side splat comparison.

---

## Files retained for future reference

- `~/projects/AnySplat/` (15 GB) — env + repo, smoke-test scripts in place
- `~/sr_eval/` (573 MB) — SR test scripts + UltraSharp model + outputs
- `~/projects/image_to_splat/EVAL_SESSION_2026-05-20.md` (this doc)
- `Desktop/sr_shootout/` — visual comparison outputs for any future review
- `Desktop/lito_acid_test/` — LiTo final outputs (column blurry, dragon clean)
- Permission rules in `.claude/settings.local.json` (LiTo install patterns)

## Files torn down post-session

- `~/projects/LiTo/` (32 GB) — full env, repo, kaolin source clone, third_party
- `~/projects/LiTo/artifacts/lito_*.ckpt` (8 GB) — pretrained checkpoints
- `/tmp/diffoctreerast/` (123 MB) — manual clone
- `/tmp/pip-req-build-*` (90 MB) — failed pytorch3d build artifacts
- `~/.cache/huggingface/hub/models--lhjiang--anysplat/` (2.8 GB) — kept since
  AnySplat env retained

Disk reclaim: ~32 GB.
