"""
Rigging stage — universal step for all four backends since none rig natively.

Mixamo's auto-rigger is web-based (upload mesh → get FBX back), not a
public REST API — Adobe doesn't publish one. Two realistic implementation
paths, pick one before wiring this up for real:

  A) Browser automation (Selenium/Playwright) driving mixamo.com's actual
     upload/rig/download flow. Fragile to Adobe's frontend changes, but
     zero-cost and matches "free auto-rigger" from the spec.
  B) Swap Mixamo for a self-hosted alternative with a real API (e.g. a
     local rigging tool/algorithm) if browser automation proves too brittle
     in production. Worth flagging back to the user if (A) breaks often.

This module is written against interface (A) since that's what the spec
names explicitly, with the automation calls stubbed out.
"""

from pathlib import Path
from dataclasses import dataclass


class MalformedRigError(Exception):
    """Raised when the returned rig fails bone-hierarchy validation."""
    pass


@dataclass
class RigResult:
    rigged_fbx_path: Path
    bone_count: int
    has_standard_hierarchy: bool


def upload_and_rig(mesh_path: Path, output_dir: Path, source_pose: str) -> RigResult:
    """
    Drives Mixamo's auto-rigger.

    source_pose matters: CharacterGen's canonical A-pose output rigs more
    reliably here than the arbitrary poses from TripoSR/InstantMesh/TRELLIS,
    per spec. If source_pose != "a_pose_canonical", expect a higher
    malformed-rig rate — this function should still attempt it and let
    validate_rig() catch failures rather than skipping non-CharacterGen output.

    TODO: implement the actual Mixamo browser-automation flow here:
      1. Upload mesh_path to mixamo.com
      2. Wait for auto-detection of hip/joints (may need manual marker
         placement fallback if auto-detect fails on a non-canonical pose)
      3. Select a neutral/no-animation rig export (we just want the skeleton
         + skin weights, not a baked animation)
      4. Download resulting FBX to output_dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rigged_path = output_dir / "mesh_rigged.fbx"

    # --- placeholder for real automation ---
    # driver = launch_browser()
    # driver.upload(mesh_path)
    # driver.wait_for_autorig()
    # driver.download_fbx(rigged_path)

    bone_count = _inspect_bone_count(rigged_path)
    has_standard = _has_standard_hierarchy(rigged_path)

    return RigResult(
        rigged_fbx_path=rigged_path,
        bone_count=bone_count,
        has_standard_hierarchy=has_standard,
    )


def validate_rig(result: RigResult) -> None:
    """
    Per spec: reject and flag malformed rigs rather than silently packaging
    a broken asset. Call this before proceeding to the upscale/package stages.
    """
    if not result.has_standard_hierarchy:
        raise MalformedRigError(
            f"Rig at {result.rigged_fbx_path} does not have a standard "
            f"root → spine → limbs hierarchy (bone_count={result.bone_count}). "
            f"Job flagged, not packaged."
        )


def _inspect_bone_count(fbx_path: Path) -> int:
    # TODO: parse FBX (e.g. via Blender's Python API `bpy` running headless,
    # or a library like `pyfbx`) and count bones in the armature.
    return 0


def _has_standard_hierarchy(fbx_path: Path) -> bool:
    # TODO: check for root -> spine -> limb chain in the parsed armature.
    # Blender headless (`blender --background --python check_rig.py`) is the
    # most reliable option since it's the same tool consuming this package
    # downstream in Phase 5.5.
    return False
