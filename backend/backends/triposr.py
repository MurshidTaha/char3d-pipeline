"""
TripoSR backend — fast draft/preview pass, single forward pass, no diffusion loop.
Repo: https://github.com/VAST-AI-Research/TripoSR
"""

import sys
import gc
from pathlib import Path
import torch
from PIL import Image
import rembg

# Dynamically add the cloned TripoSR repo to Python's path so it can be imported
triposr_path = Path("/kaggle/working/char3d-pipeline/TripoSR")
if str(triposr_path) not in sys.path:
    sys.path.append(str(triposr_path))

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
        
        from tsr.system import TSR
        
        print("Loading TripoSR weights to GPU...")
        self._model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        # Lower chunk size prevents VRAM spikes on the T4 GPU during rendering
        self._model.renderer.set_chunk_size(8192)
        self._model.to("cuda:0")
        
        self._loaded = True

    def unload(self):
        if self._model is not None:
            print("Unloading TripoSR and freeing VRAM...")
            del self._model
            self._model = None
            
            # Force garbage collection to clear out the VRAM for the next backend
            gc.collect()
            torch.cuda.empty_cache()
            
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        from tsr.utils import remove_background, resize_foreground
        
        print(f"Loading reference image: {reference_image_path}")
        image = Image.open(reference_image_path).convert("RGBA")
        
        # TripoSR requires a cleanly cropped foreground with no background
        print("Removing background and centering subject...")
        session = rembg.new_session()
        image = remove_background(image, session)
        image = resize_foreground(image, 0.85)

        print("Running TripoSR 3D generation...")
        with torch.no_grad():
            scene_codes = self._model([image], device="cuda:0")
            meshes = self._model.extract_mesh(scene_codes)
            mesh = meshes[0]

        mesh_path = output_dir / "raw_mesh.obj"
        print(f"Exporting raw mesh to: {mesh_path}")
        mesh.export(mesh_path)

        return {
            "mesh_path": mesh_path,
            "texture_paths": [],  # TripoSR uses vertex colors, not separate PBR texture maps
            "pose": "arbitrary_input_pose",  
        }