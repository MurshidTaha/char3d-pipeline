"""
InstantMesh backend — general organic shapes (animals, fantasy creatures) where
CharacterGen's anime-specific training doesn't fit.
Repo: https://github.com/TencentARC/InstantMesh

Sensitive to input image perspective, per spec — needs a clean, front-facing
reference image. Consider validating/cropping the reference image before
calling generate() if you see poor results in practice.
"""

from pathlib import Path
from .base import Backend3DGenerator


class InstantMeshBackend(Backend3DGenerator):
    name = "instantmesh"
    vram_gb_estimate = 16

    def __init__(self):
        super().__init__()
        self._model = None

    def load(self):
        if self._loaded:
            return
        # self._model = load_instantmesh_pipeline(...)
        self._loaded = True

    def unload(self):
        if self._model is not None:
            # del self._model; torch.cuda.empty_cache()
            self._model = None
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        # multiview = self._model.generate_multiview(reference_image_path)
        # mesh, texture = self._model.reconstruct(multiview)

        mesh_path = output_dir / "raw_mesh.glb"
        texture_path = output_dir / "raw_basecolor.png"
        # mesh.export(mesh_path); texture.save(texture_path)

        return {
            "mesh_path": mesh_path,
            "texture_paths": [texture_path],
            "pose": "arbitrary_input_pose",  # does NOT auto-canonicalize, per spec
        }
