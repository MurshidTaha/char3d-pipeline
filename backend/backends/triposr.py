"""
TripoSR backend — fast draft/preview pass, single forward pass, no diffusion loop.
Repo: https://github.com/VAST-AI-Research/TripoSR

Lowest VRAM of the four (~6-8GB) — comfortable headroom on a T4 even alongside
other light processes. Used both as the routing fallback and as the forced
backend when "Quick Preview" is toggled in the UI.
"""

from pathlib import Path
from .base import Backend3DGenerator


class TripoSRBackend(Backend3DGenerator):
    name = "triposr"
    vram_gb_estimate = 7

    def __init__(self):
        super().__init__()
        self._model = None

    def load(self):
        if self._loaded:
            return
        # self._model = TSR.from_pretrained("stabilityai/TripoSR", ...)
        self._loaded = True

    def unload(self):
        if self._model is not None:
            # del self._model; torch.cuda.empty_cache()
            self._model = None
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        # scene_codes = self._model([reference_image_path], device="cuda")
        # mesh = self._model.extract_mesh(scene_codes)[0]

        mesh_path = output_dir / "raw_mesh.obj"
        # mesh.export(mesh_path)

        return {
            "mesh_path": mesh_path,
            "texture_paths": [],  # TripoSR bakes vertex color / simple texture, not separate PBR maps
            "pose": "arbitrary_input_pose",  # does NOT auto-canonicalize, per spec
        }
