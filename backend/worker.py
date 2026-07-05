"""
FIFO job queue + worker loop.

Per spec: this tool shares a T4 with Phase 5 SDXL work, so jobs queue rather
than run in parallel. A simple Python list + worker loop is enough at current
scale — no need for Celery/Redis/etc.
"""

import threading
import queue
import traceback
from pathlib import Path

from .models import Job, JobStatus
from .config import select_backend
from .backends import backend_manager
from . import rigging, upscaling, packaging, cache

STORAGE_DIR = Path(__file__).parent / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"
JOBS: dict[str, Job] = {}  # in-memory job store; fine at this scale, single process

_job_queue: "queue.Queue[str]" = queue.Queue()
_worker_thread = None
_worker_lock = threading.Lock()


def enqueue_job(job: Job) -> None:
    JOBS[job.job_id] = job
    _job_queue.put(job.job_id)
    _ensure_worker_running()


def _ensure_worker_running():
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
            _worker_thread.start()


def _worker_loop():
    while True:
        job_id = _job_queue.get()  # blocks until a job is available
        job = JOBS.get(job_id)
        if job is None:
            continue
        try:
            _process_job(job)
        except Exception as e:
            job.status = JobStatus.failed
            job.error = f"{e}\n{traceback.format_exc()}"
            job.log_line(f"❌ Failed: {e}")
        finally:
            _job_queue.task_done()


def _process_job(job: Job):
    req = job.request
    work_dir = JOBS_DIR / job.job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    job.log_line(f"Job received for '{req.character_name}' ({req.organism_type.value}/{req.style_mode.value})")

    # --- Backend selection ---
    override = "triposr" if req.quick_preview else (
        req.backend_override.value if req.backend_override else None
    )
    backend_name, reason = select_backend(req.organism_type.value, req.style_mode.value, override)
    job.backend_used = backend_name
    job.routing_reason = reason
    job.log_line(f"Routed to backend '{backend_name}' — {reason}")

    # --- Cache check ---
    ref_image_path = Path(job.reference_image_path)
    image_hash = cache.hash_image(ref_image_path)
    key = cache.cache_key(image_hash, req.organism_type.value, req.style_mode.value, backend_name)
    cached_zip = cache.lookup(key)
    if cached_zip:
        job.cache_hit = True
        job.output_zip_path = cached_zip
        job.status = JobStatus.complete
        job.log_line("✅ Cache hit — reusing previously generated package, skipping full pipeline")
        return

    # --- Stage 1: Generation ---
    job.status = JobStatus.running_generation
    job.log_line(f"Loading {backend_name} model weights (first job on a fresh server may take a few minutes)...")
    backend = backend_manager.get(backend_name)
    job.log_line(f"{backend_name} loaded — running multi-view generation on reference image...")
    gen_dir = work_dir / "generation"
    gen_result = backend.run_job(ref_image_path, gen_dir, req.pose_hint.value)
    job.log_line("Generation stage complete — mesh produced. Unloading model to free VRAM...")
    backend_manager.unload_current()  # free VRAM immediately, per spec

    # --- Stage 2: Rigging ---
    job.status = JobStatus.running_rigging
    job.log_line("Running Blender auto-rigger (skeleton fit + automatic weight skinning)...")
    rig_dir = work_dir / "rigging"
    rig_result = rigging.upload_and_rig(
        gen_result["mesh_path"], rig_dir, source_pose=gen_result["pose"]
    )
    rigging.validate_rig(rig_result)  # raises MalformedRigError -> job marked failed, nothing packaged
    job.log_line(f"Rig complete — {rig_result.bone_count} bones, standard hierarchy confirmed")

    # --- Stage 3: Upscaling ---
    job.status = JobStatus.running_upscale
    job.log_line("Upscaling textures...")
    upscale_dir = work_dir / "upscaled"
    upscaled_textures = upscaling.upscale_textures(gen_result["texture_paths"], upscale_dir)
    job.log_line("Texture upscaling complete")

    # --- Stage 4 + 5: Packaging ---
    job.status = JobStatus.packaging
    job.log_line("Packaging final .glb, .fbx, textures, and metadata into zip...")
    metadata = packaging.build_metadata(job, backend_name, reason)
    zip_path = packaging.build_package(
        character_name=req.character_name,
        work_dir=work_dir,
        rigged_fbx_path=rig_result.rigged_fbx_path,
        upscaled_textures=upscaled_textures,
        reference_image_path=ref_image_path,
        metadata=metadata,
    )

    cache.store(key, str(zip_path))
    job.output_zip_path = str(zip_path)

    # package_dir (work_dir/character_name) still has the loose mesh.glb on
    # disk — shutil.make_archive() zips it but doesn't delete the source
    # folder. Point directly at it so the frontend's embedded 3D viewer can
    # fetch the raw glb without unzipping anything client-side.
    glb_candidate = work_dir / req.character_name / "mesh.glb"
    if glb_candidate.exists():
        job.output_glb_path = str(glb_candidate)

    job.status = JobStatus.complete
    job.log_line(f"✅ Complete — {req.character_name}.zip ready for download")


def get_job(job_id: str) -> Job | None:
    return JOBS.get(job_id)


def queue_position(job_id: str) -> int:
    """Rough position estimate — good enough for progress-polling UI, not exact."""
    pending = list(_job_queue.queue)
    if job_id in pending:
        return pending.index(job_id) + 1
    return 0
