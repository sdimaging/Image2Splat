"""
Backbone abstraction over TRELLIS.2 (and Pixal3D once installed).

Both microsoft/TRELLIS.2 and TencentARC/Pixal3D expose the same
ImageTo3DPipeline.from_pretrained(...).run(image) API surface, so a single
class can route either via a string flag.

The TRELLIS.2 codebase is not pip-installable upstream — it expects you to run
scripts from inside the cloned repo. We sys.path-inject the repo root on first
import so a thin wrapper from anywhere can `from src.backbone import Backbone`.

Usage:
    from src.backbone import Backbone
    bb = Backbone(name="trellis2", model="microsoft/TRELLIS.2-4B").load()
    mesh = bb.run(PIL.Image.open("photo.png"), seed=42)
    # mesh.vertices, mesh.faces, mesh.attrs, mesh.coords, mesh.layout, mesh.voxel_size
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from PIL import Image

# ---------------------------------------------------------------------------
# sys.path injection
# ---------------------------------------------------------------------------

DEFAULT_TRELLIS2_REPO = Path.home() / "projects" / "TRELLIS.2"


def _inject_repo_path(repo_dir: Path) -> None:
    """Add an upstream repo root to sys.path so its top-level module imports work.

    No-op if already on sys.path. Raises if the directory doesn't look like a clone.
    """
    repo_dir = repo_dir.resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"backbone repo not found at {repo_dir}")
    path_str = str(repo_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


DEFAULT_PIXAL3D_REPO = Path.home() / "projects" / "Pixal3D"


# Pixal3D's four image-conditioning model configs. These attach as attributes
# on the pipeline AFTER from_pretrained loads the base weights. The DinoV3 path
# uses camenduru's ungated mirror — sidesteps Meta's gate that vanilla TRELLIS.2 hits.
PIXAL3D_IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# MoGe-2 monocular geometry model — used by Pixal3D to estimate camera params
# (camera_angle_x, distance) from the input image for pixel-aligned back-projection.
MOGE_MODEL = "Ruicheng/moge-2-vitl"


@dataclass
class Backbone:
    """Wraps a TRELLIS.2-style image-to-3D pipeline behind a stable API.

    Two backbones supported:
      - "trellis2": vanilla microsoft/TRELLIS.2-4B
      - "pixal3d":  TencentARC/Pixal3D — built on TRELLIS.2 with pixel-aligned
                    back-projection. Better visible-side fidelity. Requires natten
                    (NAF upsampler), MoGe (camera estimation), and uses 4 DinoV3
                    image-cond models attached to the pipeline post-load.

    Attributes:
        name: which backbone to use. "trellis2" or "pixal3d".
        model: HuggingFace repo ID or local path for the pipeline weights.
        device: torch device string, default "cuda".
        repo_dir: path to the cloned upstream repo. Auto-defaults to the right one
                  per backbone if not specified.
        max_num_tokens: inference budget; higher = sharper but more VRAM.
        pixal3d_pipeline_type: which Pixal3D pipeline mode to run. "1024_cascade"
                               is the upstream default — 1024-token shape SLat.
    """

    name: str = "trellis2"
    model: str = ""  # auto-set based on `name` if empty
    device: str = "cuda"
    repo_dir: Optional[Path] = None  # auto-set based on `name` if None
    max_num_tokens: int = 49152
    pixal3d_pipeline_type: str = "1024_cascade"
    # Optional dicts merged into the pipeline's default sampler params.
    # Common keys: {"steps": N, "guidance_strength": F}. Bumping steps refines diffusion;
    # higher guidance leans harder on the input image conditioning.
    # NOTE: Pixal3D's sampler.sample() takes `guidance_strength`, not `cfg_strength`.
    # Per-sampler defaults from pipeline.json (verified 2026-05-20):
    #   sparse_structure: steps=12, guidance_strength=7.5, guidance_interval=[0.6,1.0], rescale_t=5.0
    #   shape_slat:       steps=12, guidance_strength=7.5, guidance_interval=[0.6,1.0], rescale_t=3.0
    #   tex_slat:         steps=12, guidance_strength=1.0, guidance_interval=[0.6,0.9], rescale_t=3.0
    # The shape stages use strong image adherence (7.5); texture is loose (1.0). DO NOT
    # override all three with the same value — destroys per-stage tuning.
    sparse_structure_sampler_params: dict = field(default_factory=dict)
    shape_slat_sampler_params: dict = field(default_factory=dict)
    tex_slat_sampler_params: dict = field(default_factory=dict)

    _pipeline: Optional[Any] = field(default=None, init=False, repr=False)
    _moge: Optional[Any] = field(default=None, init=False, repr=False)

    SUPPORTED_BACKBONES = ("trellis2", "pixal3d")

    def __post_init__(self) -> None:
        if self.name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"unknown backbone {self.name!r}; supported: {self.SUPPORTED_BACKBONES}"
            )
        if not self.model:
            self.model = (
                "microsoft/TRELLIS.2-4B" if self.name == "trellis2" else "TencentARC/Pixal3D"
            )
        if self.repo_dir is None:
            self.repo_dir = (
                DEFAULT_TRELLIS2_REPO if self.name == "trellis2" else DEFAULT_PIXAL3D_REPO
            )
        self.repo_dir = Path(self.repo_dir).expanduser()

    def load(self) -> "Backbone":
        """Load the pipeline from the repo + pretrained weights. Idempotent."""
        if self._pipeline is not None:
            return self

        # EXR support for HDRI envmap
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        # Reduce fragmentation on the 5090's 32GB
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if self.name == "trellis2":
            _inject_repo_path(self.repo_dir)
            from trellis2.pipelines import Trellis2ImageTo3DPipeline

            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(self.model)
            pipeline.to(self.device)
            self._pipeline = pipeline

        elif self.name == "pixal3d":
            # Pixal3D's reference inference.py uses flash_attn 3, but that requires
            # building Dao-AILab/flash-attention v3 from source (separate from v2 PyPI).
            # We have flash-attn 2.8.2 installed, which Pixal3D's attention.full_attn
            # supports via the 'flash_attn' backend. Set BEFORE importing pixal3d.
            os.environ.setdefault("ATTN_BACKEND", "flash_attn")
            _inject_repo_path(self.repo_dir)

            from pixal3d.pipelines import Pixal3DImageTo3DPipeline
            from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
                DinoV3ProjFeatureExtractor,
            )

            pipeline = Pixal3DImageTo3DPipeline.from_pretrained(self.model)

            # Pixal3D pipeline needs 4 image-cond models attached post-load
            pipeline.image_cond_model_ss = DinoV3ProjFeatureExtractor(
                **PIXAL3D_IMAGE_COND_CONFIGS["ss"]
            )
            pipeline.image_cond_model_shape_512 = DinoV3ProjFeatureExtractor(
                **PIXAL3D_IMAGE_COND_CONFIGS["shape_512"]
            )
            pipeline.image_cond_model_shape_1024 = DinoV3ProjFeatureExtractor(
                **PIXAL3D_IMAGE_COND_CONFIGS["shape_1024"]
            )
            pipeline.image_cond_model_tex_1024 = DinoV3ProjFeatureExtractor(
                **PIXAL3D_IMAGE_COND_CONFIGS["tex_1024"]
            )
            for m in (
                pipeline.image_cond_model_ss,
                pipeline.image_cond_model_shape_512,
                pipeline.image_cond_model_shape_1024,
                pipeline.image_cond_model_tex_1024,
            ):
                m.eval()

            # Pixal3D's low_vram mode offloads idle cond models to CPU. Without
            # this, 4× DINOv3 + NAF + pipeline weights + cutlass-fna attention
            # buffers exceed the 5090's 32GB at 1024-cascade resolution.
            pipeline.low_vram = True
            pipeline.cuda()
            for attr in (
                "image_cond_model_ss",
                "image_cond_model_shape_512",
                "image_cond_model_shape_1024",
                "image_cond_model_tex_1024",
            ):
                m = getattr(pipeline, attr, None)
                if m is not None:
                    m.cuda()
                    if getattr(m, "use_naf_upsample", False):
                        # NAF (valeoai/NAF) is loaded via torch.hub. Its hubconf.py
                        # does `from src.model.naf import NAF`, but NAF's `src/` has
                        # NO __init__.py (namespace package), while our project's
                        # `src/` IS a regular package. Python's importlib prefers
                        # regular packages over namespace packages — even when the
                        # namespace one is earlier in sys.path. So our `src/`
                        # always wins, breaking the NAF import.
                        #
                        # Fix: temporarily remove our project root from sys.path AND
                        # clear `src*` from sys.modules. Then the only candidate
                        # for `src` is NAF's namespace package. Restore everything
                        # after.
                        import sys as _sys  # noqa: WPS433
                        proj_root = str(Path(__file__).resolve().parent.parent)
                        saved_paths = [p for p in _sys.path if p == proj_root]
                        _sys.path[:] = [p for p in _sys.path if p != proj_root]
                        saved_src_module = _sys.modules.pop("src", None)
                        saved_src_submodules = {
                            k: v for k, v in list(_sys.modules.items())
                            if k == "src" or k.startswith("src.")
                        }
                        for k in saved_src_submodules:
                            _sys.modules.pop(k, None)
                        try:
                            m._load_naf()
                        finally:
                            # Restore project root + our src module bindings
                            for p in saved_paths:
                                if p not in _sys.path:
                                    _sys.path.insert(0, p)
                            if saved_src_module is not None:
                                _sys.modules["src"] = saved_src_module
                            for k, v in saved_src_submodules.items():
                                _sys.modules[k] = v

            self._pipeline = pipeline

            # MoGe is loaded lazily on first .run() since it's only used for cam estimation
        else:
            raise AssertionError("unreachable")
        return self

    def _ensure_moge(self):
        """Load MoGe-2 monocular-geometry model (Pixal3D only, lazy)."""
        if self._moge is not None:
            return self._moge
        from moge.model.v2 import MoGeModel  # noqa: WPS433

        self._moge = MoGeModel.from_pretrained(MOGE_MODEL).to(self.device).eval()
        return self._moge

    def _estimate_camera_params(self, image, mesh_scale: float = 1.0,
                                 image_resolution: int = 512) -> dict:
        """Use MoGe to estimate (camera_angle_x, distance, mesh_scale) from a PIL image.

        Replicates Pixal3D's inference.py:get_camera_params_wild_moge logic, but skips
        the temp-file round-trip — operates on the in-memory PIL image directly.
        """
        import math
        import numpy as np
        import torch

        moge = self._ensure_moge()
        pil_image = image.convert("RGB")
        width, _ = pil_image.size
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(self.device)
        with torch.no_grad():
            output = moge.infer(image_tensor)
        intrinsics = output["intrinsics"].squeeze().cpu().numpy()
        fx = float(intrinsics[0, 0] * width)
        camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))

        # Pixal3D's distance_from_fov math, inlined
        rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        gp = np.array([-1.0, 0.0, 0.0]) @ rot.T
        gp = gp / mesh_scale / 2.0
        xw, yw, _zw = float(gp[0]), float(gp[1]), float(gp[2])
        xt, yt = 0.0, image_resolution - 1.0
        focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
        f_pixels = focal_length * image_resolution / 32.0
        x_ndc = xt - image_resolution / 2.0
        # y_ndc = -(yt - image_resolution / 2.0)  # unused in this formula path
        distance = f_pixels * xw / x_ndc - yw
        return {
            "camera_angle_x": float(camera_angle_x),
            "distance": float(distance),
            "mesh_scale": float(mesh_scale),
        }

    def run(
        self,
        image: "Image.Image",
        seed: int = 42,
        preprocess: bool = True,
    ):
        """Run image-to-3D inference and return the first mesh.

        For TRELLIS.2: standard pipeline.run(image).
        For Pixal3D:   estimate camera_params via MoGe, then pipeline.run(image, camera_params=...).

        Returns a MeshWithVoxel — we do NOT call .simplify() so our renderer can
        preserve fine detail in the multi-view sampling.
        """
        if self._pipeline is None:
            raise RuntimeError("backbone not loaded; call .load() first")

        if self.name == "trellis2":
            meshes = self._pipeline.run(
                image,
                num_samples=1,
                seed=seed,
                preprocess_image=preprocess,
                max_num_tokens=self.max_num_tokens,
            )
        elif self.name == "pixal3d":
            # Pixal3D needs the preprocessed image FIRST so MoGe runs on the same
            # alpha-masked / center-cropped image the diffusion pipeline will see.
            if preprocess:
                image_for_inference = self._pipeline.preprocess_image(image)
            else:
                image_for_inference = image
            camera_params = self._estimate_camera_params(image_for_inference)
            result = self._pipeline.run(
                image_for_inference,
                camera_params=camera_params,
                num_samples=1,
                seed=seed,
                preprocess_image=False,  # we already preprocessed above
                return_latent=False,
                pipeline_type=self.pixal3d_pipeline_type,
                max_num_tokens=self.max_num_tokens,
                sparse_structure_sampler_params=self.sparse_structure_sampler_params,
                shape_slat_sampler_params=self.shape_slat_sampler_params,
                tex_slat_sampler_params=self.tex_slat_sampler_params,
            )
            # Pixal3D returns either a List[Mesh] (return_latent=False) or
            # (List[Mesh], (shape_slat, tex_slat, res)) (return_latent=True).
            meshes = result if isinstance(result, list) else result[0]
        else:
            raise AssertionError("unreachable")

        if not meshes:
            raise RuntimeError("pipeline returned no meshes")
        return meshes[0]

    @property
    def pipeline(self):
        """Direct access to the underlying pipeline object (for advanced calls)."""
        if self._pipeline is None:
            raise RuntimeError("backbone not loaded; call .load() first")
        return self._pipeline
