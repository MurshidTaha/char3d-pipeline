"""
Character-to-3D Tool — FastAPI backend.

Run on Kaggle T4 behind ngrok, same pattern as your Phase 1/6 setup:

    uvicorn backend.main:app --host 0.0.0.0 --port 8000
    # then: ngrok http 8000

Endpoints:
    POST /jobs                 submit a single generation job (multipart: image + form fields)
    POST /jobs/batch           submit N jobs (per spec: queued sequentially, not parallel)
    GET  /jobs/{job_id}        poll status
    GET  /jobs/{job_id}/download  download the finished .zip
    GET  /config/routing       inspect current routing table
    PUT  /config/routing       update routing table without redeploying (per spec)
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import Job, GenerationRequest, OrganismType, StyleMode, PoseHint, BackendChoice
from . import worker
from .config import load_routing_table, save_routing_table

app = FastAPI(title="Character-to-3D Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this once the frontend is served from a fixed origin
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = Path(__file__).parent / "storage" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(job_id: str, upload: UploadFile) -> Path:
    dest = UPLOADS_DIR / f"{job_id}_{upload.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@app.post("/jobs")
async def submit_job(
    character_name: str = Form(...),
    character_description: str = Form(...),
    organism_type: OrganismType = Form(...),
    style_mode: StyleMode = Form(...),
    pose_hint: PoseHint = Form(PoseHint.auto),
    backend_override: BackendChoice | None = Form(None),
    quick_preview: bool = Form(False),
    reference_image: UploadFile = File(...),
):
    req = GenerationRequest(
        character_name=character_name,
        character_description=character_description,
        organism_type=organism_type,
        style_mode=style_mode,
        pose_hint=pose_hint,
        backend_override=backend_override,
        quick_preview=quick_preview,
    )
    job = Job(request=req)
    job.reference_image_path = str(_save_upload(job.job_id, reference_image))

    worker.enqueue_job(job)
    return {"job_id": job.job_id, "status": job.status}


@app.post("/jobs/batch")
async def submit_batch(
    character_names: list[str] = Form(...),
    character_descriptions: list[str] = Form(...),
    organism_types: list[OrganismType] = Form(...),
    style_modes: list[StyleMode] = Form(...),
    reference_images: list[UploadFile] = File(...),
):
    """
    Batch endpoint — e.g. generate your whole cast (Aryan, Zara, Dadi, Biscuit,
    Mango, Mr. Khan) in one call. Per spec, this queues N jobs sequentially
    given the T4-sharing constraint — it does not run them in parallel.
    """
    lengths = {
        len(character_names), len(character_descriptions),
        len(organism_types), len(style_modes), len(reference_images),
    }
    if len(lengths) != 1:
        raise HTTPException(400, "All batch fields must have the same length (one entry per character).")

    job_ids = []
    for name, desc, organism, style, image in zip(
        character_names, character_descriptions, organism_types, style_modes, reference_images
    ):
        req = GenerationRequest(
            character_name=name,
            character_description=desc,
            organism_type=organism,
            style_mode=style,
        )
        job = Job(request=req)
        job.reference_image_path = str(_save_upload(job.job_id, image))
        worker.enqueue_job(job)
        job_ids.append(job.job_id)

    return {"job_ids": job_ids}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = worker.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    resp = job.model_dump()
    if job.status == "queued":
        resp["queue_position"] = worker.queue_position(job_id)
    return resp


@app.get("/jobs/{job_id}/download")
async def download_job(job_id: str):
    job = worker.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "complete" or not job.output_zip_path:
        raise HTTPException(409, f"Job not ready (status: {job.status})")
    return FileResponse(
        job.output_zip_path,
        media_type="application/zip",
        filename=f"{job.request.character_name}.zip",
    )


@app.get("/config/routing")
async def get_routing_table():
    return load_routing_table()


@app.put("/config/routing")
async def update_routing_table(table: dict):
    save_routing_table(table)
    return {"status": "updated", "table": table}


@app.get("/health")
async def health():
    return {"status": "ok"}
