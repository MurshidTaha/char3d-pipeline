"""
TRELLIS backend — real implementation, built from microsoft/TRELLIS's own
example.py. Of the three previously-stubbed backends, this one has by far
the cleanest published API (a real installable `trellis` package with
`TrellisImageTo3DPipeline.from_pretrained()` / `.run()`), so there's much
less reverse-engineering risk here than CharacterGen or InstantMesh.

IMPORTANT PREREQUISITE (not yet done by Cell 1): the actual TRELLIS repo
must be cloned to /kaggle/working/char3d-pipeline/TRELLIS, and its setup.sh
run to build native extensions. This is the heaviest install of the four
backends — TRELLIS depends on compiled CUDA extensions (spconv, nvdiffrast,
diffoctreerast, kaolin, mip-splatting, and flash-attn OR xformers for
attention) that setup.sh builds from source. Add to Cell 1:

    TRELLIS_URL = "https://github.com/microsoft/TRELLIS.git"
    TRELLIS_DIR = f"{REPO_DIR}/TRELLIS"
    if not os.path.exists(f"{TRELLIS_DIR}/setup.sh"):
        run(["git", "clone", "--recurse-submodules", TRELLIS_URL, TRELLIS_DIR])
    # T4 doesn't support flash-attn — use xformers instead (see load() below,
    # which sets ATTN_BACKEND=xformers to match this install):
    run(["bash", "./setup.sh", "--basic", "--xformers", "--diffoctreerast",
         "--spconv", "--mipgaussian", "--kaolin", "--nvdiffrast"], cwd=TRELLIS_DIR)

KNOWN RISK #1: TRELLIS's own hardware notes say it's verified on A100/A6000
(24-40GB+) and lists "at least 16GB" as the floor. A T4 is exactly 16GB, and
that's before accounting for whatever else is resident — this is the
single most likely backend to OOM on your hardware, not a bug to "fix" in
this file so much as a genuine hardware ceiling to watch for in logs.

KNOWN RISK #2: flash-attn (the repo's default attention backend) generally
doesn't build/run on T4s (they're pre-Ampere, no proper flash-attn support)
— hence forcing ATTN_BACKEND=xformers below, which setup.sh's --xformers
flag needs to have installed at build time.
"""

import os
# Must be set before `import trellis` — read once at process/module import,
# not something that can be configured after the fact.
os.environ.setdefault("ATTN_BACKEND", "xformers")   # flash-attn doesn't support T4s
os.environ.setdefault("SPCONV_ALGO", "native")      # 'auto' benchmarks on first call; 'native' is fine for single-shot jobs

import sys
import gc
from pathlib import Path

import torch
from PIL import Image

from .base import Backend3DGenerator

REPO_ROOT = Path("/kaggle/working/char3d-pipeline")
TRELLIS_DIR = REPO_ROOT / "TRELLIS"

if str(TRELLIS_DIR) not in sys.path:
    sys.path.append(str(TRELLIS_DIR))


class TRELLISBackend(Backend3DGenerator):
    name = "trellis"
    vram_gb_estimate = 16

    def __init__(self):
        super().__init__()
        self._pipeline = None

    def load(self):
        if self._loaded:
            return

        if not TRELLIS_DIR.exists():
            raise RuntimeError(
                f"TRELLIS repo not found at {TRELLIS_DIR} — clone it and run "
                f"its setup.sh in Cell 1 before this backend can run (see "
                f"module docstring)."
            )

        from trellis.pipelines import TrellisImageTo3DPipeline

        print("Loading TRELLIS weights (microsoft/TRELLIS-image-large)...")
        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS-image-large"
        )
        self._pipeline.cuda()
        self._loaded = True

    def unload(self):
        if self._pipeline is not None:
            print("Unloading TRELLIS and freeing VRAM...")
            del self._pipeline
            self._pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)

        from trellis.utils import postprocessing_utils

        print(f"Running TRELLIS on {reference_image_path}")
        image = Image.open(reference_image_path).convert("RGB")

        # TRELLIS does its own background/foreground handling internally —
        # unlike TripoSR/InstantMesh, it doesn't need a separate rembg pass
        # before this call.
        outputs = self._pipeline.run(image, seed=1)

        print("Baking Gaussian splat + mesh into one textured GLB (postprocessing_utils.to_glb)...")
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,      # mesh simplification ratio — matches TRELLIS's own example
            texture_size=1024,
        )
        mesh_path = output_dir / "raw_mesh.glb"
        glb.export(str(mesh_path))
        print(f"Exported: {mesh_path}")

        return {
            "mesh_path": mesh_path,
            # to_glb() bakes a full baked texture directly into the GLB's
            # embedded material — no separate side-car texture file, same
            # convention as TripoSR's vertex-color approach.
            "texture_paths": [],
            "pose": "arbitrary_input_pose",
        }