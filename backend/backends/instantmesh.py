import sys
import gc
import torch
from pathlib import Path
from .base import Backend3DGenerator

# Dynamically add InstantMesh to path
repo_path = Path("/kaggle/working/char3d-pipeline/InstantMesh")
if str(repo_path) not in sys.path:
    sys.path.append(str(repo_path))

class InstantMeshBackend(Backend3DGenerator):
    name = "instantmesh"
    vram_gb_estimate = 16

    def __init__(self):
        super().__init__()
        self._model = None

    def load(self):
        if self._loaded:
            return
        
        # NOTE: Replace with actual InstantMesh loader
        print("Loading InstantMesh weights...")
        # self._model = load_instantmesh_pipeline(...)
        self._loaded = True

    def unload(self):
        if self._model is not None:
            print("Unloading InstantMesh and freeing VRAM...")
            del self._model
            self._model = None
            gc.collect()
            torch.cuda.empty_cache()
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running InstantMesh on {reference_image_path}")
        
        # multiview = self._model.generate_multiview(reference_image_path)
        # mesh, texture = self._model.reconstruct(multiview)

        mesh_path = output_dir / "raw_mesh.glb"
        texture_path = output_dir / "raw_basecolor.png"
        
        print(f"Exporting raw mesh to: {mesh_path}")
        # mesh.export(mesh_path)
        # texture.save(texture_path)

        return {
            "mesh_path": mesh_path,
            "texture_paths": [texture_path],
            "pose": "arbitrary_input_pose",
        }