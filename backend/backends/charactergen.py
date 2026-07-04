import sys
import gc
import torch
from pathlib import Path
from .base import Backend3DGenerator

# Dynamically add CharacterGen to path
repo_path = Path("/kaggle/working/char3d-pipeline/CharacterGen")
if str(repo_path) not in sys.path:
    sys.path.append(str(repo_path))

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
        
        # NOTE: Replace 'load_stage2d_pipeline' with actual import from CharacterGen repo
        # from charactergen_module import load_stage2d_pipeline
        print("Loading CharacterGen 2D weights...")
        # self._stage2d_model = load_stage2d_pipeline(weights_only=True)
        self._loaded = True

    def _load_stage3d(self):
        self._unload_stage2d()
        
        # NOTE: Replace 'load_stage3d_pipeline' with actual import
        print("Loading CharacterGen 3D weights...")
        # self._stage3d_model = load_stage3d_pipeline(weights_only=True)

    def _unload_stage2d(self):
        if self._stage2d_model is not None:
            print("Unloading CharacterGen 2D weights...")
            del self._stage2d_model
            self._stage2d_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def unload(self):
        self._unload_stage2d()
        if self._stage3d_model is not None:
            print("Unloading CharacterGen 3D weights...")
            del self._stage3d_model
            self._stage3d_model = None
            gc.collect()
            torch.cuda.empty_cache()
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running CharacterGen on {reference_image_path}")

        # Stage 2D
        # multiview_images = self._stage2d_model.run(reference_image_path, pose_hint=pose_hint)

        # Stage 3D
        self._load_stage3d()
        # mesh, texture = self._stage3d_model.reconstruct(multiview_images)

        mesh_path = output_dir / "raw_mesh.glb"
        texture_path = output_dir / "raw_basecolor.png"
        
        # Execute the actual export!
        print(f"Exporting raw mesh to: {mesh_path}")
        # mesh.export(mesh_path)
        # texture.save(texture_path)

        return {
            "mesh_path": mesh_path,
            "texture_paths": [texture_path],
            "pose": "a_pose_canonical",
        }