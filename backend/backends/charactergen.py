"""
CharacterGen backend — real implementation wired to zjp-shadow/CharacterGen's
own webui.py classes (Inference2D_API, Inference3D_API, rm_bg_api), rather
than reimplementing their pipeline.

IMPORTANT PREREQUISITE (not yet done by Cell 1): the actual CharacterGen repo
must be cloned to /kaggle/working/char3d-pipeline/CharacterGen, and its OWN
requirements.txt installed, before this module can import anything. TripoSR
is already cloned by Cell 1; CharacterGen is not. Add to Cell 1:

    CHARGEN_URL = "https://github.com/zjp-shadow/CharacterGen.git"
    CHARGEN_DIR = f"{REPO_DIR}/CharacterGen"
    if not os.path.exists(f"{CHARGEN_DIR}/webui.py"):
        run(["git", "clone", CHARGEN_URL, CHARGEN_DIR])
    run(["pip", "install", "-q", "--break-system-packages",
         "-r", f"{CHARGEN_DIR}/requirements.txt"])

KNOWN RISK: CharacterGen's README specifies Python 3.9. This Kaggle env is
Python 3.12 (per the cp312 wheel names from Cell 1's torchmcubes build).
Their requirements.txt pins an old diffusers/accelerate/transformers stack
built around Tune-A-Video — there is a real chance this collides with the
numpy2/rembg-driven dependency resolution from Cell 1, the same way the
scikit-image/opencv pins did earlier. If `import webui` fails at the
`check_min_version("0.24.0")` line or on `tuneavideo`/`lrm` imports, that's
the dependency graph fighting itself again, not a bug in this file — paste
the traceback and we resolve it the same way as before (let the resolver
settle, don't force old pins).
"""

import sys
import os
import gc
import shutil
from pathlib import Path

import torch

from .base import Backend3DGenerator

REPO_ROOT = Path("/kaggle/working/char3d-pipeline")
CHARGEN_DIR = REPO_ROOT / "CharacterGen"

for p in (CHARGEN_DIR, CHARGEN_DIR / "2D_Stage", CHARGEN_DIR / "3D_Stage"):
    if str(p) not in sys.path:
        sys.path.append(str(p))


class CharacterGenBackend(Backend3DGenerator):
    name = "charactergen"
    vram_gb_estimate = 16

    def __init__(self):
        super().__init__()
        self._cg = None          # the imported webui module itself
        self._rmbg = None        # webui.rm_bg_api instance
        self._infer2d = None     # webui.Inference2D_API instance
        self._infer3d = None     # webui.Inference3D_API instance (loaded lazily, per job)

    def load(self):
        if self._loaded:
            return

        if not CHARGEN_DIR.exists():
            raise RuntimeError(
                f"CharacterGen repo not found at {CHARGEN_DIR} — clone it in "
                f"Cell 1 before this backend can run (see module docstring)."
            )

        # webui.py's own code uses paths relative to the repo root
        # (e.g. "./2D_Stage/configs/infer.yaml"), so this process must be
        # cwd'd there for both the import and every call below.
        os.chdir(CHARGEN_DIR)

        from omegaconf import OmegaConf
        import webui as cg  # noqa: this import runs their weight-download
                             # check as a side effect (idempotent — skips
                             # files that already exist locally); main() is
                             # guarded by __name__ == "__main__" so importing
                             # this does NOT launch their Gradio server.
        self._cg = cg

        print("Loading CharacterGen background-removal model (rm_bg_api)...")
        self._rmbg = cg.rm_bg_api()

        print("Loading CharacterGen 2D stage weights (Inference2D_API)...")
        cfg2d = OmegaConf.load("./2D_Stage/configs/infer.yaml")
        self._infer2d = cg.Inference2D_API(**cfg2d)

        self._loaded = True

    def _unload_2d(self):
        for attr in ("_infer2d", "_rmbg"):
            obj = getattr(self, attr)
            if obj is not None:
                print(f"Unloading CharacterGen {attr.lstrip('_')}...")
                del obj
                setattr(self, attr, None)
        gc.collect()
        torch.cuda.empty_cache()

    def _load_stage3d(self):
        if self._infer3d is not None:
            return
        print("Loading CharacterGen 3D stage weights (Inference3D_API)...")
        self._infer3d = self._cg.Inference3D_API()

    def unload(self):
        self._unload_2d()
        if self._infer3d is not None:
            print("Unloading CharacterGen 3D stage...")
            del self._infer3d
            self._infer3d = None
            gc.collect()
            torch.cuda.empty_cache()
        self._loaded = False

    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(CHARGEN_DIR)  # generate() can be called well after load(); re-assert cwd

        from PIL import Image
        print(f"Running CharacterGen 2D multi-view stage on {reference_image_path}")
        image = Image.open(reference_image_path).convert("RGBA")

        # Matches webui.py's gen4views(): clean the input's background first.
        image = self._rmbg.remove_background([image], alpha_min=0.1, alpha_max=0.9)[0]

        # Inference2D_API.inference() returns 4 raw multi-view renders.
        # 512x768, seed, and timestep=40 mirror webui.py's own UI defaults.
        raw_views = self._infer2d.inference(
            image, 512, 768, crop=True, seed=2333, timestep=40
        )
        masked_views = self._rmbg.remove_background(raw_views, alpha_min=0.2, alpha_max=0.9)

        # webui.py's button1.click maps the 4 returned images positionally to
        # outputs=[img_input2, img_input0, img_input3, img_input1], i.e. the
        # inference() return order is [Right, Back, Left, Front]. Reorder
        # to the (Back, Front, Right, Left) order Inference3D_API expects
        # (matches button2.click's inputs=[img_input0..3] wiring exactly).
        right, back, left, front = masked_views

        # Free the 2D stage's VRAM before loading the 3D stage — both together
        # likely exceed a T4's 16GB (this backend's own vram_gb_estimate),
        # mirroring the original stub's intent.
        self._unload_2d()
        self._load_stage3d()

        print("Running CharacterGen 3D reconstruction stage...")
        save_dir, obj_path, glb_path = self._infer3d.process_images(
            back, front, right, left, back_proj=False, smooth_iter=5,
        )

        mesh_path = output_dir / "raw_mesh.glb"
        shutil.copy(glb_path, mesh_path)
        print(f"Copied CharacterGen output to: {mesh_path}")

        return {
            "mesh_path": mesh_path,
            # CharacterGen bakes its texture into vertex colors during
            # webui.py's traverse() step before export — same convention as
            # TripoSR, no separate PBR texture map to hand off.
            "texture_paths": [],
            "pose": "a_pose_canonical",
        }