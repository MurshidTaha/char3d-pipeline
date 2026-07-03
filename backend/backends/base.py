"""
Common interface all four backends implement.

IMPORTANT: These are stubs with the correct load/unload/infer lifecycle,
job semantics, and file contracts wired up. The actual model inference calls
(torch.hub / from_pretrained / repo-specific loaders) need to be filled in
against whichever exact checkpoint/repo revision you pin, since those APIs
shift between commits. Each stub tells you exactly what to swap in.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class Backend3DGenerator(ABC):
    name: str = "base"
    vram_gb_estimate: int = 0

    def __init__(self):
        self._loaded = False

    @abstractmethod
    def load(self):
        """Load model weights onto GPU. Must be idempotent-safe (no-op if already loaded)."""
        ...

    @abstractmethod
    def unload(self):
        """Free VRAM. Called after every job — per spec, only one backend fits on a T4 at a time."""
        ...

    @abstractmethod
    def generate(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        """
        Run inference on reference_image_path, write raw mesh + texture into output_dir.

        Returns dict with at least:
            {
                "mesh_path": Path,      # native export, GLB or OBJ depending on backend
                "texture_paths": list[Path],
                "pose": str,            # actual pose of output mesh — matters for rigging step
            }
        """
        ...

    def run_job(self, reference_image_path: Path, output_dir: Path, pose_hint: str) -> dict:
        """Convenience wrapper: load → generate → leave loaded (caller decides when to unload)."""
        if not self._loaded:
            self.load()
        return self.generate(reference_image_path, output_dir, pose_hint)
