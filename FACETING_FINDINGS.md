# Metal faceting investigation — findings (2026-06-11)

## Symptom
~90% of metal probe cells (helms/swords/armor) show large angular bright/dark
plateaus on smooth surfaces. 2-3 seeds out of 20 come out clean.

## Root cause (numerically confirmed)
**The generated GEOMETRY itself undulates in plateau-scale tilted patches.**
Channel-isolation diagnostic on a worst-case breastplate (T5 seed222, view 129):

| channel                  | bright vs dark plateau delta |
|--------------------------|------------------------------|
| shaded luminance         | 1.80× ratio (the artifact)   |
| base_color               | +1.5% (clean)                |
| roughness                | -3.5% (clean)                |
| metallic                 | +4.9% (clean)                |
| AO (clay)                | +14.9% (secondary contributor)|
| **surface normal angle** | **19.2° between plateaus**   |

Metal + HDRI turns 19° orientation differences into 1.8× brightness jumps.
Seed-dependence: some shape-SLat samples decode smooth panels, most decode
"hand-forged bumpy" panels. NOT a texture problem, NOT flat shading.

## What was tried
1. **Smooth Phong vertex normals** (pbr_mesh_renderer.py patch) — correct fix
   for flat shading, verified working (normal channel smooth), but plateaus
   are real geometry so faithful normals still show them. KEEP (it helps
   decimation-related faceting and costs nothing).
2. **Crease-aware normal-field diffusion** (25 iters, 35° crease,
   env PIXAL3D_NORMAL_SMOOTH_ITERS / _CREASE in the same patch) —
   measured 1.80→1.76 ratio, 19.2°→15.5°. INSUFFICIENT at feasible
   iteration counts (diffusion radius too small vs plateau scale).

## Fix 3: SSAO intensity 1.5 → 0.8 (SHIPPED 2026-06-11)
One-line change to `pbr_mesh_renderer.py`; default now env-var overridable
via `PIXAL3D_SSAO_INTENSITY` (default 0.8).

**A/B measurement on same mesh, same view (Chest.png T5 seed222 view129):**

| | clay_fg (higher = less AO darkening) |
|---|---|
| SSAO=1.5 (old) | 0.783 — 21.7% light blocked |
| SSAO=0.8 (new) | 0.883 — 11.7% light blocked |

Result: **AO contribution cut in half.** The plateau contrast ratio is reduced
proportionally. Geometry plateaus still exist; this is a partial fix.

Patch updated: `patches/pixal3d_smooth_normals.patch` includes the SSAO change.
A/B composite images: `/tmp/facet_diag_ab_chest/AB_comparison.png`

## Fix 4: plateau geometry band-stop — EXPERIMENTAL, default OFF (2026-06-11)

**STATUS CORRECTION (same day):** initially shipped default-ON; flipped to
default-OFF after measurement. The filter softens plateaus (breastplate
lum ratio 2.06 -> 1.91 at full strength) but ADDS high-frequency
normal-field noise (+40-70% Laplacian energy, measured), concentrated at
feature edges (flutes/straps/rivets — heatmap-localized). Root tension:
flutes live at the same WAVELENGTH as plateaus; only amplitude separates
them, and every gate strong enough to protect features also weakens the
plateau correction (v11: ratio 2.06 -> 2.01, still +41% HF). For a splat
pipeline this is disqualifying — PostShot bakes the noise in from 200
views. 11 design iterations are documented in mesh_smooth.py's docstring;
the module stays for experimentation via `--geo-smooth` (opt-in).

Practical alternatives, in order: seed auto-ranking (plateau-metric
scoring of probe cells), Hunyuan 3.1 mesh comparison (clean-topology API
meshes may not facet at all), SDF-level smoothing inside the decode.

<details><summary>Original (over-optimistic) ship notes</summary>

`scripts/mesh_smooth.py::plateau_smooth` — surface-aware removal of the
plateau-band undulation from the mesh itself, post-decimation, <1s on 7.5M
verts. Wired into the daemon as `--geo-smooth` (default ON) in BOTH the
production and probe paths; banner line shows state.

Architecture (8 iterations of design — failed approaches documented in the
module docstring so nobody retries them):
1. Vertex clustering (1.6% grid cells + 6-way normal bin) -> centroids.
2. Cluster GRAPH from mesh edges (surface connectivity — stacked surfaces
   never mix; verified zero displacement on parallel planes 5% apart).
3. Implicit low-pass `(I+tL)^2 x = c` via conjugate gradient (reaches
   plateau wavelengths in one solve) + band self-cleanup (squares the
   large-scale leak).
4. Triplanar-blended trilinear interpolation of the band back to vertices
   (kills cluster-border steps AND bin-seam silhouette notching).
5. Safety stack: soft amplitude gate (0.5% bbox tanh), normal-coherence
   gate min-capped by own cluster (suppresses rims/teeth/detail), open-
   boundary freeze with faded dilation, and a LOCAL SHELL-THICKNESS cap
   (35% of own-to-opposite-bin distance — prevents thin-shell z-fighting).

Verified on three assets (Mimic chest, ornate knight piece, fluted
breastplate): plateau patches visibly softened in shaded + normal channels,
silhouettes clean, teeth/straps/flutes/engraving intact, no texture loss.
Breastplate metrics: lum p90/p10 2.06 -> 1.91, std 0.107 -> 0.098.
Evidence: docs/faceting_ab/. The correction is conservative by design
(thickness cap binds on thin shells) — partial fix, never harmful.

Companion renderer change: `attr_vertices` texture anchor in
pbr_mesh_renderer.py (voxel attrs sampled at PRE-displacement positions —
without it, moved verts read outside the sparse attr shell = black).
In patches/pixal3d_smooth_normals.patch.

Tooling: `diagnose_facets.py --geo-compare` (A/B on one inference) and
`--mesh-cache` (torch.save the mesh; render-only iteration ~40s vs ~9min).

</details>

## Remaining candidate steps
1. **SDF/voxel-grid smoothing** before mesh extraction (attacks root even
   earlier; needs TRELLIS decode internals).
2. **Seed auto-selection**: score probe cells by plateau metric (large-scale
   luminance std on metal masks) and auto-rank — automates the current
   manual 2-3-good-out-of-20 workflow.
3. Raise the shell-thickness budget (0.35 -> ~0.45) if z-fighting stays
   absent in production; recovers correction strength on thin armor.
4. Longer term: SLat-PiD H100 training (see Desktop/Image2Splat_Research/).

## Tooling
- `scripts/diagnose_facets.py` — renders ONE asset cell, dumps ALL renderer
  channels (shaded/base_color/roughness/metallic/normal/clay) for channel
  isolation. Plus the numerical plateau measurement used above (in git log
  of this file / session notes).
- `patches/pixal3d_smooth_normals.patch` — current full local Pixal3D diff
  (Phong normals + optional diffusion + "(proj)" labels).
