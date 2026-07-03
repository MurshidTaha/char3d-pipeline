"""
Upscaling stage.

Texture upscale reuses your existing Real-ESRGAN setup from Phase 1 (per spec —
don't reinvent this, just point it at the new texture maps). Mesh
remesh/subdivision is optional and only relevant for close-up shot use.
"""

from pathlib import Path


def upscale_textures(texture_paths: list[Path], output_dir: Path, target: str = "4k") -> list[Path]:
    """
    Runs each texture through Real-ESRGAN.

    TODO: import your existing Phase 1 Real-ESRGAN wrapper here instead of
    reimplementing — e.g.:
        from phase1.upscale import run_realesrgan
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    upscaled = []
    for tex_path in texture_paths:
        out_name = f"{tex_path.stem}_{target}{tex_path.suffix}"
        out_path = output_dir / out_name
        # run_realesrgan(input_path=tex_path, output_path=out_path, scale=4)
        upscaled.append(out_path)
    return upscaled


def remesh_if_needed(mesh_path: Path, output_dir: Path, force: bool = False) -> Path:
    """
    Optional subdivision/remesh pass for low-poly backend output when the
    target use is close-up shots (per spec). Off by default — only run when
    the caller explicitly requests it, since it adds real processing time.
    """
    if not force:
        return mesh_path

    output_dir.mkdir(parents=True, exist_ok=True)
    remeshed_path = output_dir / f"{mesh_path.stem}_remeshed{mesh_path.suffix}"
    # TODO: Blender headless subdivision surface modifier pass, e.g.:
    #   blender --background --python remesh.py -- --in mesh_path --out remeshed_path
    return remeshed_path
