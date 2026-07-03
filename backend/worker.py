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
            job.touch()
        finally:
            _job_queue.task_done()


def _process_job(job: Job):
    req = job.request
    work_dir = JOBS_DIR / job.job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Backend selection ---
    override = "triposr" if req.quick_preview else (
        req.backend_override.value if req.backend_override else None
    )
    backend_name, reason = select_backend(req.organism_type.value, req.style_mode.value, override)
    job.backend_used = backend_name
    job.routing_reason = reason
    job.touch()

    # --- Cache check ---
    ref_image_path = Path(job.reference_image_path)
    image_hash = cache.hash_image(ref_image_path)
    key = cache.cache_key(image_hash, req.organism_type.value, req.style_mode.value, backend_name)
    cached_zip = cache.lookup(key)
    if cached_zip:
        job.cache_hit = True
        job.output_zip_path = cached_zip
        job.status = JobStatus.complete
        job.touch()
        return

    # --- Stage 1: Generation ---
    job.status = JobStatus.running_generation
    job.touch()
    backend = backend_manager.get(backend_name)
    gen_dir = work_dir / "generation"
    gen_result = backend.run_job(ref_image_path, gen_dir, req.pose_hint.value)
    backend_manager.unload_current()  # free VRAM immediately, per spec

    # --- Stage 2: Rigging ---
    job.status = JobStatus.running_rigging
    job.touch()
    rig_dir = work_dir / "rigging"
    rig_result = rigging.upload_and_rig(
        gen_result["mesh_path"], rig_dir, source_pose=gen_result["pose"]
    )
    rigging.validate_rig(rig_result)  # raises MalformedRigError -> job marked failed, nothing packaged

    # --- Stage 3: Upscaling ---
    job.status = JobStatus.running_upscale
    job.touch()
    upscale_dir = work_dir / "upscaled"
    upscaled_textures = upscaling.upscale_textures(gen_result["texture_paths"], upscale_dir)

    # --- Stage 4 + 5: Packaging ---
    job.status = JobStatus.packaging
    job.touch()
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
    job.status = JobStatus.complete
    job.touch()


def get_job(job_id: str) -> Job | None:
    return JOBS.get(job_id)


def queue_position(job_id: str) -> int:
    """Rough position estimate — good enough for progress-polling UI, not exact."""
    pending = list(_job_queue.queue)
    if job_id in pending:
        return pending.index(job_id) + 1
    return 0
