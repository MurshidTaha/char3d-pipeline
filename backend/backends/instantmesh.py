"""
InstantMesh backend — real implementation, built directly from
TencentARC/InstantMesh's own run.py (Stage 1: Zero123++ multiview diffusion,
Stage 2: LRM triplane reconstruction + mesh extraction), not reimplemented
from scratch.

IMPORTANT PREREQUISITE (not yet done by Cell 1): the actual InstantMesh repo
must be cloned to /kaggle/working/char3d-pipeline/InstantMesh, and its OWN
requirements.txt installed. Add to Cell 1:

    INSTANTMESH_URL = "https://github.com/TencentARC/InstantMesh.git"
    INSTANTMESH_DIR = f"{REPO_DIR}/InstantMesh"
    if not os.path.exists(f"{INSTANTMESH_DIR}/run.py"):
        run(["git", "clone", INSTANTMESH_URL, INSTANTMESH_DIR])
    run(["pip", "install", "-q", "--break-system-packages",
         "-r", f"{INSTANTMESH_DIR}/requirements.txt"])

The repo also ships its config files (configs/instant-mesh-large.yaml etc.)
— those come along with the git clone, nothing extra to fetch there.

KNOWN RISK: InstantMesh's README recommends torch>=2.1.0 + xformers built
against a matching CUDA — a version-pinned combo, same category of fragility
as CharacterGen's. If `torch`/`xformers` mismatch on import, that's the
dependency graph again, not this file.
"""

import os
import sys
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torchvision.transforms import v2
from omegaconf import OmegaConf

from .base import Backend3DGenerator

REPO_ROOT = Path("/kaggle/working/char3d-pipeline")
INSTANTMESH_DIR = REPO_ROOT / "InstantMesh"

if str(INSTANTMESH_DIR) not in sys.path:
    sys.path.append(str(INSTANTMESH_DIR))

CONFIG_NAME = "instant-mesh-large"  # matches configs/instant-mesh-large.yaml — IS_FLEXICUBES=True


class InstantMeshBackend(Backend3DGenerator):
    name = "instantmesh"
    vram_gb_estimate = 16

    def __init__(self):
        super().__init__()
        self._diffusion_pipeline = None   # Zero123++ (Stage 1, multiview)
        self._recon_model = None          # LRM triplane reconstructor (Stage 2)
        self._infer_config = None
        self._device = torch.device("cuda")

    def load(self):
        if self._loaded:
            return

        if not INSTANTMESH_DIR.exists():
            raise RuntimeError(
                f"InstantMesh repo not found at {INSTANTMESH_DIR} — clone it "
                f"in Cell 1 before this backend can run (see module docstring)."
            )

        # Everything below mirrors run.py's Stage 0 config load + Stage 1
        # diffusion pipeline setup exactly — same repo IDs, same custom
        # white-background UNet swap.
        os.chdir(INSTANTMESH_DIR)
        config = OmegaConf.load(f"configs/{CONFIG_NAME}.yaml")
        self._infer_config = config.infer_config

        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        from huggingface_hub import hf_hub_download

        print("Loading Zero123++ multiview diffusion pipeline...")
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        print("Loading InstantMesh's custom white-background UNet...")
        if os.path.exists(self._infer_config.unet_path):
            unet_ckpt_path = self._infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename="diffusion_pytorch_model.bin",
                repo_type="model",
            )
        state_dict = torch.load(unet_ckpt_path, map_location="cpu")
        pipeline.unet.load_state_dict(state_dict, strict=True)
        self._diffusion_pipeline = pipeline.to(self._device)

        self._loaded = True

    def _load_recon_model(self):
        if self._recon_model is not None:
            return

        os.chdir(INSTANTMESH_DIR)
        config = OmegaConf.load(f"configs/{CONFIG_NAME}.yaml")
        model_config = config.model_config

        from src.utils.train_util import instantiate_from_config
        from huggingface_hub import hf_hub_download

        print("Loading InstantMesh LRM reconstruction model...")
        model = instantiate_from_config(model_config)
        if os.path.exists(self._infer_config.model_path):
            model_ckpt_path = self._infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename=f"{CONFIG_NAME.replace('-', '_')}.ckpt",
                repo_type="model",
            )
        state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self._device)

        # instant-mesh-* configs use the flexicubes geometry head (vs.
        # instant-nerf-* which don't) — required before extract_mesh() works.
        model.init_flexicubes_geometry(self._device, fovy=30.0)
        self._recon_model = model.eval()

    def _unload_diffusion(self):
        if self._diffusion_pipeline is not None:
            print("Unloading Zero123++ diffusion pipeline to free VRAM for reconstruction...")
            del self._diffusion_pipeline
            self._diffusion_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()

    def unload(self):
        self._unload_diffusion()
        if self._recon_model is not None:
            print("Unloading InstantMesh reconstruction model...")
            del self._recon_model
            self._recon_model = None
            gc.collect()
            torch.cuda.empty_cache()
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(INSTANTMESH_DIR)

        from src.utils.infer_util import remove_background, resize_foreground
        import rembg

        print(f"Running InstantMesh Stage 1 (multiview) on {reference_image_path}")
        input_image = Image.open(reference_image_path)
        rembg_session = rembg.new_session()
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)

        # Zero123++ returns ONE image: a 3x2 grid (960x640) packing 6 views.
        output_image = self._diffusion_pipeline(input_image, num_inference_steps=75).images[0]

        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, "c (n h) (m w) -> (n m) c h w", n=3, m=2)  # (6, 3, 320, 320)

        # Stage 1 is diffusion-heavy; free it before loading the LRM so both
        # don't have to coexist in VRAM on a single T4.
        self._unload_diffusion()
        self._load_recon_model()

        print("Running InstantMesh Stage 2 (LRM reconstruction)...")
        from src.utils.camera_util import get_zero123plus_input_cameras
        from src.utils.mesh_util import save_obj

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self._device)
        images = images.unsqueeze(0).to(self._device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        with torch.no_grad():
            planes = self._recon_model.forward_planes(images, input_cameras)
            vertices, faces, vertex_colors = self._recon_model.extract_mesh(
                planes, use_texture_map=False, **self._infer_config
            )

        obj_path = output_dir / "raw_mesh.obj"
        save_obj(vertices, faces, vertex_colors, str(obj_path))
        print(f"Exported raw mesh (OBJ, vertex colors): {obj_path}")

        # Re-export as GLB to match the "raw_mesh.glb" contract every other
        # backend uses (rigging.py/packaging.py expect one consistent
        # extension) — trimesh preserves OBJ vertex colors in the GLB export.
        import trimesh
        mesh_path = output_dir / "raw_mesh.glb"
        trimesh.load(obj_path, process=False).export(mesh_path)
        print(f"Converted to: {mesh_path}")

        return {
            "mesh_path": mesh_path,
            "texture_paths": [],  # vertex-color mesh, no separate PBR texture map
            "pose": "arbitrary_input_pose",
        }