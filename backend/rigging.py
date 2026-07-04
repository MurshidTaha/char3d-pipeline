"""
Rigging stage — universal step for all four backends since none rig natively.

SELF-HOSTED, NOT MIXAMO. Mixamo has no public REST API — its only automation
path is browser-driven (Selenium/Playwright against mixamo.com), which the
original spec already flagged as the fragile option. This implementation goes
straight to the fallback noted below instead: a self-hosted, CPU-only
auto-rigger built on Blender headless, since Blender is already the tool
consuming this package downstream in Phase 5.5 anyway.

Why this fits your box (14.6GB usable T4 VRAM, 29GB CPU RAM):
  - Skeleton fitting + skinning run entirely on CPU. Zero VRAM usage — this
    stage never competes with whichever of the 4 generation backends or your
    Phase 5 SDXL process currently owns the T4.
  - A single character mesh is a few MB; Blender's own overhead is modest.
    Comfortably inside 29GB even running alongside other CPU work.
  - Runs as a subprocess (not an in-process `bpy` import), so a Blender crash
    on a malformed mesh can't take down the FastAPI worker process, and you
    don't need Blender's Python ABI to match whatever Python runs the API.

See backend/blender_scripts/auto_rig.py for the actual rigging logic
(geometry-driven skeleton placement + Blender's automatic weight skinning).
That script writes a JSON "report" sidecar — bone count, hierarchy check,
whether any fallback heuristic was used — computed inside Blender itself,
since Blender already has the bone graph in memory. Nothing here re-parses
the exported FBX in Python.

Setup: see SETUP_GUIDE.md § "Installing Blender on Kaggle" for getting a
portable Blender build onto the notebook (no full desktop install needed).
"""

from pathlib import Path
from dataclasses import dataclass, field
import json
import subprocess
import shutil
import os

BLENDER_SCRIPT = Path(__file__).parent / "blender_scripts" / "auto_rig.py"

# Resolution order: explicit env var (set this in your Kaggle notebook after
# extracting portable Blender) -> "blender" on PATH -> the conventional
# extraction path from the SETUP_GUIDE instructions.
BLENDER_EXECUTABLE = (
    os.environ.get("BLENDER_EXECUTABLE")
    or shutil.which("blender")
    or "/kaggle/working/blender/blender"
)

RIG_TIMEOUT_SECONDS = 300  # a single character mesh should rig in well under this


class MalformedRigError(Exception):
    """Raised when a rig was produced but fails bone-hierarchy validation."""
    pass


class RiggingToolError(Exception):
    """
    Raised when the rigging *tool* itself couldn't run — missing Blender
    binary, unreadable/empty input mesh, Blender crash, timeout. Distinct
    from MalformedRigError so worker.py's error message points at the right
    problem (environment/upstream-stage vs. actual geometry).
    """
    pass


@dataclass
class RigResult:
    rigged_fbx_path: Path
    bone_count: int
    has_standard_hierarchy: bool
    bone_names: list = field(default_factory=list)
    used_fallback_leg_split: bool = False
    used_fallback_arm_pose: bool = False


def _resolve_blender() -> str:
    if shutil.which(BLENDER_EXECUTABLE):
        return BLENDER_EXECUTABLE
    if Path(BLENDER_EXECUTABLE).exists():
        return BLENDER_EXECUTABLE
    raise RiggingToolError(
        f"Blender executable not found (looked for '{BLENDER_EXECUTABLE}'). "
        f"Set the BLENDER_EXECUTABLE env var to your portable Blender binary, "
        f"or see SETUP_GUIDE.md § 'Installing Blender on Kaggle'."
    )


def upload_and_rig(mesh_path: Path, output_dir: Path, source_pose: str) -> RigResult:
    """
    Runs the self-hosted Blender auto-rigger on mesh_path.

    source_pose is threaded through unchanged from the generation stage —
    CharacterGen's canonical A-pose output is still the best-behaved input
    (arms held clear of the torso helps the shoulder/hand geometry heuristic
    in auto_rig.py), matching what the spec expected from Mixamo too.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rigged_path = output_dir / "mesh_rigged.fbx"
    sidecar_path = output_dir / "rig_report.json"

    mesh_path = Path(mesh_path)
    if not mesh_path.exists() or mesh_path.stat().st_size == 0:
        raise RiggingToolError(
            f"No usable mesh at {mesh_path} — the generation stage hasn't "
            f"produced a real mesh file yet (check that backend's generate() "
            f"isn't still a stub; mesh.export(...) needs to actually run)."
        )

    blender_bin = _resolve_blender()

    cmd = [
        blender_bin,
        "--background",
        "--factory-startup",
        "--python", str(BLENDER_SCRIPT),
        "--",
        "--input", str(mesh_path),
        "--output", str(rigged_path),
        "--report", str(sidecar_path),
        "--pose", source_pose,
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=RIG_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        raise RiggingToolError(
            f"Blender auto-rig timed out after {RIG_TIMEOUT_SECONDS}s on {mesh_path}."
        )

    if not sidecar_path.exists():
        raise RiggingToolError(
            f"Blender exited (code {proc.returncode}) without writing a rig "
            f"report — likely crashed before reaching auto_rig.py's own error "
            f"handling. stderr tail:\n{proc.stderr[-2000:]}"
        )

    report = json.loads(sidecar_path.read_text())

    if report.get("error"):
        raise RiggingToolError(f"auto_rig.py failed on {mesh_path}: {report['error']}")

    return RigResult(
        rigged_fbx_path=rigged_path,
        bone_count=report["bone_count"],
        has_standard_hierarchy=report["has_standard_hierarchy"],
        bone_names=report.get("bone_names", []),
        used_fallback_leg_split=report.get("used_fallback_leg_split", False),
        used_fallback_arm_pose=report.get("used_fallback_arm_pose", False),
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
