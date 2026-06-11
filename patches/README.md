# Local patches to upstream clones

These diffs capture local modifications to upstream repos that the pipeline
depends on but that live OUTSIDE this repository. If you re-clone the
upstream, re-apply with:

    cd ~/projects/Pixal3D && git apply /path/to/Image2Splat/patches/pixal3d_smooth_normals.patch

## pixal3d_smooth_normals.patch

Target: TencentARC/Pixal3D clone (`~/projects/Pixal3D`)

Key change — `pixal3d/renderers/pbr_mesh_renderer.py`:
Smooth (Phong) vertex normals instead of flat per-face normals in the PBR
mesh renderer. Upstream computes one normal per triangle (edge cross
product) and interpolates that constant across the face — i.e. flat
shading. After QEM decimation (which strips low-curvature regions down to
few LARGE triangles to stay under the 16.78M-face nvdiffrast cap), flat
shading turns every big triangle into a visible lighting plateau —
severe faceting on smooth glossy surfaces (metal helms, blades, domes),
amplified by HDRI specular response.

Fix: area-weighted accumulation of unnormalized face cross-products into
per-vertex normals, normalized per vertex, interpolated via the real face
index buffer, renormalized per pixel. QEM keeps real creases densely
triangulated, so smoothing does not visibly round hard edges.

The patch file also includes two small pre-existing local mods to the
pipeline ("(proj)" sampler labelling) captured for completeness.

Applied: 2026-06-11.
