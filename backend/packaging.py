"""
Packaging stage — assembles the final download structure and zips it.

Output structure (per spec):
    character_name/
    ├── mesh.glb
    ├── mesh.fbx
    ├── textures/
    │   ├── basecolor_4k.png
    │   ├── normal_4k.png
    │   ├── roughness_4k.png
    ├── reference_render.png
    └── metadata.json
"""

import json
import shutil
import time
from pathlib import Path


def build_package(
    character_name: str,
    work_dir: Path,
    rigged_fbx_path: Path,
    upscaled_textures: list[Path],
    reference_image_path: Path,
    metadata: dict,
) -> Path:
    """
    Assembles the package directory and returns the path to the final .zip.
    """
    package_dir = work_dir / character_name
    textures_dir = package_dir / "textures"
    textures_dir.mkdir(parents=True, exist_ok=True)

    # mesh.fbx — the rigged mesh, primary animation-pipeline format
    shutil.copy2(rigged_fbx_path, package_dir / "mesh.fbx")

    # mesh.glb — spec lists this as the primary format too; if the rig step
    # only produced FBX, note that a GLB re-export (e.g. via Blender headless)
    # is needed here. Left as an explicit TODO rather than silently skipped.
    glb_target = package_dir / "mesh.glb"
    if rigged_fbx_path.with_suffix(".glb").exists():
        shutil.copy2(rigged_fbx_path.with_suffix(".glb"), glb_target)
    # else: TODO — Blender headless FBX->GLB export before packaging

    # textures — rename into the basecolor/normal/roughness_4k.png convention
    for tex_path in upscaled_textures:
        dest_name = _classify_texture_name(tex_path)
        shutil.copy2(tex_path, textures_dir / dest_name)

    # reference_render.png — quick-look thumbnail
    if reference_image_path and reference_image_path.exists():
        shutil.copy2(reference_image_path, package_dir / "reference_render.png")

    # metadata.json — reproducibility/audit trail, per spec ("matters more than it looks")
    with open(package_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    zip_path = work_dir / f"{character_name}.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", package_dir)
    return zip_path


def _classify_texture_name(tex_path: Path) -> str:
    """Best-effort classification of which PBR channel a texture file is, based on filename hints."""
    stem = tex_path.stem.lower()
    if "normal" in stem:
        return "normal_4k.png"
    if "rough" in stem:
        return "roughness_4k.png"
    return "basecolor_4k.png"


def build_metadata(job, backend_used: str, routing_reason: str) -> dict:
    return {
        "character_name": job.request.character_name,
        "prompt": job.request.character_description,
        "organism_type": job.request.organism_type,
        "style_mode": job.request.style_mode,
        "pose_hint_requested": job.request.pose_hint,
        "backend_used": backend_used,
        "routing_reason": routing_reason,
        "generation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": job.job_id,
        # TODO: thread the actual seed through from the backend's generate() call
        # once real inference is wired up — needed for true regeneration/audit trail.
        "seed": None,
    }
