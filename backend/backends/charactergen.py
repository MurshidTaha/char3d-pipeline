"""
CharacterGen backend.

Repo: https://github.com/zjpshadow/CharacterGen
Weights: https://huggingface.co/zjpshadow/CharacterGen  (Apache 2.0, confirmed — see config.py BACKEND_INFO)

Two-stage pipeline, per spec:
  1. 2D Stage: pose canonicalization + multi-view diffusion
  2. 3D Stage: sparse-view reconstruction

These are two separate model loads. Per spec's VRAM note: unload the 2D stage
weights before loading the 3D stage if VRAM is tight (T4 16GB, and this backend
alone wants ~16GB at each stage).

TODO before first real run:
  - Pin an exact commit/revision of the HF repo.
  - One checkpoint file is flagged "scanned unsafe" on the model card — load with
    torch.load(..., weights_only=True) where the loader supports it, per spec.
  - Confirm actual entrypoint function names against the pinned commit (they are
    not hardcoded here since the repo's inference API may change).
"""

from pathlib import Path
from .base import Backend3DGenerator


class CharacterGenBackend(Backend3DGenerator):
    name = "charactergen"
    vram_gb_estimate = 16

    def __init__(self):
        super().__init__()
        self._stage2d_model = None
        self._stage3d_model = None

    def load(self):
        if self._loaded:
            return
        # --- Stage 2D: pose canonicalization + multi-view diffusion ---
        # self._stage2d_model = load_stage2d_pipeline(weights_only=True)
        self._loaded = True

    def _load_stage3d(self):
        # Free stage-2D weights first if VRAM is tight (per spec)
        self._unload_stage2d()
        # self._stage3d_model = load_stage3d_pipeline(weights_only=True)

    def _unload_stage2d(self):
        if self._stage2d_model is not None:
            # del self._stage2d_model; torch.cuda.empty_cache()
            self._stage2d_model = None

    def unload(self):
        self._unload_stage2d()
        if self._stage3d_model is not None:
            # del self._stage3d_model; torch.cuda.empty_cache()
            self._stage3d_model = None
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 2D: canonicalize pose + generate multi-view images
        # multiview_images = self._stage2d_model.run(reference_image_path, pose_hint=pose_hint)

        # Stage 3D: sparse-view reconstruction from the multi-view set
        self._load_stage3d()
        # mesh, texture = self._stage3d_model.reconstruct(multiview_images)

        mesh_path = output_dir / "raw_mesh.glb"
        texture_path = output_dir / "raw_basecolor.png"
        # mesh.export(mesh_path); texture.save(texture_path)

        return {
            "mesh_path": mesh_path,
            "texture_paths": [texture_path],
            # CharacterGen canonicalizes to a known A-pose — this is *why* the spec
            # says its output rigs more reliably in Mixamo than the other 3 backends.
            "pose": "a_pose_canonical",
        }
