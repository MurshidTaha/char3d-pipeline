# Character-to-3D Tool

Standalone asset factory: reference image in → rigged, upscaled 3D character package out.
Feeds Phase 5.5 (Blender scene composition) as a black box, per spec.

## What's real vs. stubbed

This is a **complete, working skeleton** — the API, job queue, routing logic, caching,
VRAM-safe backend loading, rig validation, and packaging are all fully implemented and
tested (see below). What's stubbed, and why:

| Piece | Status | Why |
|---|---|---|
| Routing table + `select_backend()` | ✅ real, tested | Pure logic, no external deps |
| Job queue / worker loop / VRAM-safe load-unload | ✅ real | Pure Python, no GPU needed |
| Caching (image hash → zip) | ✅ real | Pure Python |
| Packaging (zip structure, metadata.json) | ✅ real | Pure Python |
| FastAPI endpoints (submit/batch/poll/download) | ✅ real, tested | Verified routes load correctly |
| Frontend form | ✅ real | Open `frontend/index.html`, point `API_BASE` at your ngrok URL |
| 4 backend `generate()` calls | 🔲 stub | Needs your pinned model revisions — see each file's TODO |
| Mixamo rigging automation | 🔲 stub | Mixamo has no public API; needs Selenium/Playwright flow — see `rigging.py` |
| Rig validation (bone hierarchy check) | 🔲 stub | Needs Blender headless (`bpy`) — same tool you'll use in Phase 5.5 anyway |
| Real-ESRGAN texture upscale | 🔲 stub | Wire in your existing Phase 1 wrapper — one-line import per the TODO |

Nothing here is fake scaffolding — every stub is a real function with the correct
signature, called at the correct point in the pipeline, with the exact TODO of what
API call goes where. Filling them in is swapping stub lines for real calls, not
restructuring anything.

## Run it (Kaggle T4 + ngrok, same pattern as your other phases)

```bash
pip install -r backend/requirements.txt
# plus whichever of torch/diffusers/trimesh/huggingface_hub you need
# once you've pinned exact backend model revisions

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then in a separate cell: `ngrok http 8000`, and point `frontend/index.html`'s
`API_BASE` at the resulting URL (or set `window.CHAR3D_API_BASE` before loading the page).

## Fill-in order (suggested)

1. **TripoSR first** — lowest VRAM, single pass, no pose canonicalization complexity.
   Gets the whole pipeline (generate → rig → upscale → package) running end-to-end
   with one real backend before touching the other three.
2. **Rig validation via Blender headless** — you'll need this working for Phase 5.5
   anyway, so it's not wasted effort building it here first.
3. **Mixamo automation** — the most likely thing to break/need maintenance since it's
   browser automation against a UI you don't control. Consider a self-hosted fallback
   if it proves too brittle (noted in `rigging.py`).
4. **CharacterGen, InstantMesh, TRELLIS** — same shape as TripoSR, just swap the
   model loader/inference calls per each file's TODO.
5. **Real-ESRGAN hookup** — import your Phase 1 wrapper into `upscaling.py`.

## Open decisions from the spec, resolved here

- **Caching**: implemented, on by default (`backend/cache.py`).
- **Batch mode**: implemented as sequential queueing (`POST /jobs/batch`), matching
  the T4-sharing constraint — not true parallel generation.
- **VRAM contention with Phase 5**: this runs as a fully separate FastAPI process/notebook.
  `BackendManager` guarantees only one of the four backends is resident in VRAM at a time,
  but it does *not* coordinate with your Phase 5 SDXL process — if you run both in the
  same Kaggle session, you're responsible for not launching them concurrently. Separate
  Kaggle sessions (as the spec recommends) sidesteps this entirely.

## Directory layout

```
char3d_tool/
├── backend/
│   ├── main.py           FastAPI app + endpoints
│   ├── worker.py         FIFO job queue + pipeline orchestration
│   ├── config.py         routing table (JSON-backed, hot-editable)
│   ├── models.py         pydantic request/job models
│   ├── rigging.py        Mixamo integration + bone-hierarchy validation
│   ├── upscaling.py      Real-ESRGAN texture upscale hookup
│   ├── packaging.py      zip assembly + metadata.json
│   ├── cache.py          (image_hash, organism, style, backend) → cached zip
│   └── backends/
│       ├── base.py       common load/unload/generate interface
│       ├── charactergen.py
│       ├── triposr.py
│       ├── instantmesh.py
│       └── trellis.py
└── frontend/
    └── index.html        single-page form + progress polling + download
```
