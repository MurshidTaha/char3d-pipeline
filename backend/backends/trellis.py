"""
TRELLIS backend — hard-surface / mechanical objects, photoreal humans, props.
Repo: https://github.com/microsoft/TRELLIS

Single-object focus, per spec — cannot segment multi-object scenes, so the
reference image must already be a clean single-subject crop (your existing
SDXL character-sheet pipeline should already produce this).
"""

from pathlib import Path
from .base import Backend3DGenerator


class TRELLISBackend(Backend3DGenerator):
    name = "trellis"
    vram_gb_estimate = 16

    def __init__(self):
        super().__init__()
        self._model = None

    def load(self):
        if self._loaded:
            return
        # self._model = load_trellis_pipeline(...)
        self._loaded = True

    def unload(self):
        if self._model is not None:
            # del self._model; torch.cuda.empty_cache()
            self._model = None
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        # outputs = self._model.run(reference_image_path)
        # mesh, texture = outputs["mesh"], outputs["texture"]

        mesh_path = output_dir / "raw_mesh.glb"
        texture_path = output_dir / "raw_basecolor.png"
        # mesh.export(mesh_path); texture.save(texture_path)

        return {
            "mesh_path": mesh_path,
            "texture_paths": [texture_path],
            "pose": "arbitrary_input_pose",  # does NOT auto-canonicalize, per spec
        }
