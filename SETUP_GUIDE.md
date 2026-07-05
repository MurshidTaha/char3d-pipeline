# Setup Guide — Character-to-3D Tool

**Version: 1.0 (fresh)** — this guide contains no session-specific values
(no ngrok URLs, no auth tokens, no personal fork paths). Everything below is
copy-paste ready for a brand-new Kaggle session; fill in only the placeholders
marked `<...>`.

Assumes the same Kaggle + ngrok + Windows-client pattern used elsewhere in
this project.

---

## 0. Prerequisites

- Kaggle account with GPU quota (T4 x1, 16GB)
- ngrok account (free tier is fine) + authtoken
- This repo, either uploaded as a Kaggle Dataset or cloned from GitHub (see
  step 1)

---

## 1. Get the code into Kaggle

**Option A — upload as a Kaggle Dataset (simplest for one-off use):**
1. Kaggle → Datasets → New Dataset → upload the repo as a zip
2. In your notebook: Add Data → your dataset
3. Unzip it into the working directory:
   ```python
   import zipfile
   zipfile.ZipFile('/kaggle/input/<your-dataset-name>/char3d-pipeline.zip').extractall('/kaggle/working/')
   %cd /kaggle/working/char3d-pipeline
   ```

**Option B — clone from GitHub (better if you'll keep editing it):**
```python
!git clone <your-repo-url>
%cd char3d-pipeline
```

---

## 2. Cell 1 — full environment setup (run once per fresh session)

This single cell installs everything: the repo's own dependencies, TripoSR
(cloned as a subfolder) plus its requirements, the compiled `torchmcubes`
extension, and the numpy-ABI rebuild that a few of those packages need to
coexist. Safe to re-run — every step is idempotent and skips work it's
already done.

```python
import os, subprocess, traceback

REPO_URL    = "<your-repo-url>"          # e.g. https://github.com/<you>/char3d-pipeline.git
REPO_DIR    = "/kaggle/working/char3d-pipeline"
TRIPOSR_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_DIR = f"{REPO_DIR}/TripoSR"

STEPS_TOTAL = 9
_n = 0
def step(t):
    global _n; _n += 1
    print(f"\n{'='*70}\n[STEP {_n}/{STEPS_TOTAL}] {t}\n{'='*70}")

def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.stdout.strip():
        print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-2500:])
        raise RuntimeError(f"Command failed (exit {r.returncode}): {' '.join(cmd)}")
    return r

try:
    step("Clone char3d-pipeline (or pull if it already exists)")
    if os.path.isdir(REPO_DIR):
        print(f"  {REPO_DIR} exists — pulling latest")
        run(["git", "pull"], cwd=REPO_DIR)
    else:
        run(["git", "clone", REPO_URL, REPO_DIR])
    print(f"  ✅ repo ready at {REPO_DIR}")

    step("Clone TripoSR (lives inside char3d-pipeline)")
    if os.path.exists(f"{TRIPOSR_DIR}/tsr/system.py"):
        print(f"  already cloned at {TRIPOSR_DIR}, skipping")
    else:
        run(["git", "clone", TRIPOSR_URL, TRIPOSR_DIR])
    print(f"  ✅ TripoSR ready at {TRIPOSR_DIR}")

    step("Install char3d-pipeline's own requirements.txt + rembg[cpu]")
    run(["pip", "install", "-q", "--break-system-packages",
         "-r", f"{REPO_DIR}/backend/requirements.txt"])
    run(["pip", "install", "-q", "--break-system-packages", "rembg[cpu]"])
    print("  ✅ base pipeline deps installed")

    step("Install TripoSR's own requirements.txt")
    triposr_reqs = f"{TRIPOSR_DIR}/requirements.txt"
    if os.path.exists(triposr_reqs):
        run(["pip", "install", "-q", "--break-system-packages", "-r", triposr_reqs])
        print("  ✅ TripoSR requirements installed")
    else:
        print(f"  ⚠️  no requirements.txt found at {triposr_reqs}, skipping")

    step("Build torchmcubes (compiled CUDA extension, ~1-3 min if missing)")
    try:
        import torchmcubes
        print(f"  ✅ already installed at {torchmcubes.__file__}, skipping build")
    except ImportError:
        os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/local/cuda/bin"
        run(["pip", "install", "-q", "--break-system-packages", "ninja"])
        run(["pip", "install", "--break-system-packages",
             "git+https://github.com/tatsy/torchmcubes.git"])
        import torchmcubes
        print(f"  ✅ built and installed: {torchmcubes.__file__}")

    step("Rebuild numpy + everything bound to its C ABI (numba, trimesh, scipy, pymatting)")
    result = subprocess.run(["pip", "show", "numpy"], capture_output=True, text=True)
    numpy_version = next(
        (line.split(":", 1)[1].strip() for line in result.stdout.splitlines()
         if line.startswith("Version:")), None
    )
    print(f"  currently resolved numpy version: {numpy_version}")

    run(["pip", "install", "-q", "--break-system-packages", "--force-reinstall",
         "--no-deps", f"numpy=={numpy_version}" if numpy_version else "numpy"])

    PINS = {"scipy": "1.13.1"}  # last scipy line that still supports numpy<2.0
    for pkg in ["numba", "trimesh", "scipy", "pymatting"]:
        if pkg in PINS:
            target = f"{pkg}=={PINS[pkg]}"
        else:
            check = subprocess.run(["pip", "show", pkg], capture_output=True, text=True)
            if check.returncode != 0:
                print(f"  ({pkg} not installed, skipping)")
                continue
            pkg_version = next(
                (line.split(":", 1)[1].strip() for line in check.stdout.splitlines()
                 if line.startswith("Version:")), None
            )
            target = f"{pkg}=={pkg_version}" if pkg_version else pkg
        run(["pip", "install", "-q", "--break-system-packages",
             "--force-reinstall", "--no-deps", target])
    print("  ✅ rebuild complete")

    step("Patch hidden dependencies (OpenCV, scikit-image, etc.) for Numpy 1.x compatibility")
    run(["pip", "install", "-q", "--break-system-packages",
         "opencv-python-headless<4.10.0", "opencv-python<4.10.0",
         "scikit-image<0.24.0", "shapely<2.0.5", "onnxruntime<1.18.0"])
    print("  ✅ hidden dependencies pinned to pre-Numpy 2.0 versions")

    step("Uninstall CuPy (broken against pinned numpy 1.26, not needed for rembg[cpu])")
    check = subprocess.run(["pip", "show", "cupy-cuda12x"], capture_output=True, text=True)
    if check.returncode == 0:
        run(["pip", "uninstall", "-y", "-q", "cupy-cuda12x"])
        print("  ✅ cupy-cuda12x removed")
    else:
        print("  (cupy-cuda12x not installed under that name, skipping)")

    step("Final verification — import everything the server will need")
    checks = {
        "torch": "torch", "torchmcubes": "torchmcubes", "rembg": "rembg",
        "trimesh": "trimesh", "PIL": "PIL", "fastapi": "fastapi", "uvicorn": "uvicorn",
    }
    all_ok = True
    for label, mod in checks.items():
        try:
            __import__(mod)
            print(f"  ✅ {label}")
        except Exception as e:
            all_ok = False
            print(f"  ❌ {label}: {type(e).__name__}: {e}")

    print(f"""
  {"✅ CELL 1 COMPLETE — everything verified, run CELL 2 next." if all_ok else "⚠️  CELL 1 finished but some imports failed above — check the output."}
""")

except Exception as e:
    print(f"\n{'!'*70}\n❌ CELL 1 FAILED AT: [STEP {_n}/{STEPS_TOTAL}]\n{'!'*70}")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()
    raise
```

Once you start filling in real backend calls for CharacterGen, InstantMesh,
or TRELLIS (see README's "Fill-in order"), add each one's extra packages as
you go rather than installing everything up front — Kaggle sessions have a
time budget and unused heavy installs just burn it.

---

## 3. Enable GPU on the notebook

Notebook settings (right panel) → Accelerator → **GPU T4 x1**. Restart the
kernel if you changed this after already running cells.

---

## 4. Set up ngrok (once, as a Kaggle Secret)

Add your authtoken as a **Kaggle Secret** named `NGROK_AUTHTOKEN` (Notebook →
Add-ons → Secrets) rather than ever pasting it into a cell in plaintext —
this keeps the notebook safe to share/publish. Cell 2 below reads it from
there automatically.

---

## 5. Cell 2 — Blender + start server + health check + ngrok tunnel

Run this after Cell 1 completes. Safe to re-run any time: it kills any stale
server on the port and reuses whatever's already installed/cached.

```python
import os, subprocess, time, socket, signal, traceback
from kaggle_secrets import UserSecretsClient

REPO_DIR        = "/kaggle/working/char3d-pipeline"
PORT            = 8000
BLENDER_VERSION = "4.2.3"  # check https://download.blender.org/release/ for current LTS
BLENDER_DIR     = f"/kaggle/working/blender-{BLENDER_VERSION}-linux-x64"

STEPS_TOTAL = 5
_n = 0
def step(t):
    global _n; _n += 1
    print(f"\n{'='*70}\n[STEP {_n}/{STEPS_TOTAL}] {t}\n{'='*70}")

def run(cmd, **kw):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.stdout.strip():
        print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-2500:])
        raise RuntimeError(f"Command failed (exit {r.returncode}): {' '.join(cmd)}")
    return r

def port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def free_port(port):
    try:
        out = subprocess.run(["fuser", f"{port}/tcp"], capture_output=True, text=True)
        for pid in out.stdout.split():
            print(f"  killing stale process on port {port}: pid {pid}")
            os.kill(int(pid), signal.SIGKILL)
        if out.stdout.split():
            time.sleep(1)
    except FileNotFoundError:
        pass

try:
    step("GPU sanity check")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True)
    print(f"  {gpu.stdout.strip() if gpu.returncode == 0 else 'WARNING: no GPU detected'}")
    if not os.path.isdir(REPO_DIR):
        raise RuntimeError(f"{REPO_DIR} missing — run CELL 1 first.")
    print(f"  ✅ repo found at {REPO_DIR}")

    step("Install / reuse portable Blender (CPU-only rigger, no VRAM used)")
    blender_bin = f"{BLENDER_DIR}/blender"
    if os.path.exists(blender_bin):
        print(f"  ✅ already installed at {blender_bin}, skipping download")
    else:
        tarball = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
        url = f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/{tarball}"
        run(["wget", "-q", url], cwd="/kaggle/working")
        run(["tar", "-xf", tarball], cwd="/kaggle/working")
        print(f"  ✅ extracted to {BLENDER_DIR}")
    if not os.path.exists(blender_bin):
        raise RuntimeError(f"Blender still missing at {blender_bin}")
    os.environ["BLENDER_EXECUTABLE"] = blender_bin
    v = subprocess.run([blender_bin, "--version"], capture_output=True, text=True)
    print(f"  {v.stdout.splitlines()[0] if v.stdout else '(no version output)'}")

    step("Start uvicorn (kills any stale server on the port first)")
    free_port(PORT)
    server = subprocess.Popen(
        ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=os.environ.copy(),
    )
    print(f"  uvicorn launched (pid {server.pid}), waiting for it to bind...")
    bound = False
    for i in range(60):
        if server.poll() is not None:
            print("  --- uvicorn's own output (exited before binding) ---")
            print(server.stdout.read())
            raise RuntimeError(f"uvicorn exited early with code {server.returncode}")
        if port_open("127.0.0.1", PORT):
            bound = True
            break
        time.sleep(0.5)
        if i % 4 == 0:
            print("  ...still waiting")
    if not bound:
        print(server.stdout.read())
        raise RuntimeError(f"uvicorn never bound to 127.0.0.1:{PORT} within 30s")
    print(f"  ✅ server listening on 127.0.0.1:{PORT}")

    step("Health check + open ngrok tunnel (forced IPv4 — avoids [::1] refused)")
    health = subprocess.run(["curl", "-s", f"http://localhost:{PORT}/health"],
                             capture_output=True, text=True)
    print(f"  /health -> {health.stdout.strip()!r}")
    if '"ok"' not in health.stdout:
        raise RuntimeError(f"/health didn't return ok: {health.stdout!r}")

    subprocess.run(["pip", "install", "-q", "pyngrok"], check=True)
    from pyngrok import ngrok
    ngrok.set_auth_token(UserSecretsClient().get_secret("NGROK_AUTHTOKEN"))
    ngrok.kill()
    public_url = ngrok.connect(addr=f"127.0.0.1:{PORT}", proto="http")
    api_base = str(public_url).split('"')[1]

    step("Done")
    print(f"""
  ✅ ALL CHECKS PASSED

  API base URL : {api_base}
  Health check : http://localhost:{PORT}/health (local) / {api_base}/health (public)
  Blender      : {blender_bin}

  Set this in your browser console (or frontend/index.html) before submitting:
    window.CHAR3D_API_BASE = "{api_base}";

  If the tunnel drops later, just re-run this cell — it cleans up and restarts.
""")

except Exception as e:
    print(f"\n{'!'*70}\n❌ CELL 2 FAILED AT: [STEP {_n}/{STEPS_TOTAL}]\n{'!'*70}")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()
    raise
```

This step is entirely CPU-side for Blender — it doesn't touch the T4, so it's
safe to run alongside any of the 3D generation backends without VRAM
contention. `rigging.py` reads `BLENDER_EXECUTABLE` from the environment, so
this cell must run in the same kernel session that then starts uvicorn
(which it does, above).

---

## 6. Point the frontend at your tunnel

On your local machine, open `frontend/index.html` — but first tell it which
API to talk to. Easiest way: add a line right before the `<script>` tag near
the bottom of `index.html`:

```html
<script>window.CHAR3D_API_BASE = "<the api_base printed by Cell 2>";</script>
```

`frontend/index.html` already checks for `window.CHAR3D_API_BASE` and falls
back to an editable placeholder if it's not set — no other file needs
changing.

---

## 7. Verify routing before running real jobs

```python
!curl http://localhost:8000/config/routing
```

This should return the JSON routing table (human/anime → charactergen, etc.).
If you ever want to re-tune a route without redeploying, `PUT` a new table to
this same endpoint.

---

## 8. Run your first job (with stubs)

At this point the full pipeline (queue → route → "generate" → "rig" →
"upscale" → package → zip) runs end-to-end, but the generation/rigging/upscale
steps are stubs that produce empty placeholder files rather than a real mesh —
this is intentional, so you can confirm the *plumbing* works before spending
time wiring in real models.

From the frontend form: fill in a character name/description, upload any
image, submit. Watch the stage strip advance and confirm you get a `.zip`
download at the end containing the expected folder structure (even with
placeholder/empty mesh files inside).

If that completes end-to-end, the API/queue/routing/packaging layer is solid
— now it's just swapping stubs for real model calls, one backend at a time
(README has the suggested fill-in order: TripoSR → rig validation via Blender
headless → the other three backends → Real-ESRGAN).

---

## 9. Polling a job and downloading the result from the command line

The frontend polls and downloads for you automatically, but you can do both
by hand too — handy for scripting or debugging without a browser. Replace
`<api-base>` with the URL Cell 2 printed (e.g. `https://xxxx.ngrok-free.dev`)
and `<job-id>` with the id returned by `POST /jobs`:

```bash
# poll status
curl -s <api-base>/jobs/<job-id>

# once status is "complete", download the finished package
curl -s -o /kaggle/working/<character-name>.zip <api-base>/jobs/<job-id>/download
```

`GET /jobs/<job-id>` also returns `queue_position` while a job is still
queued, so you can poll it in a loop until `status` flips to `complete` (or
`failed`, in which case the same response includes an `error` field).

---

## 10. Keeping the tunnel alive across a long session

ngrok free-tier tunnels can drop on inactivity or Kaggle session limits. If a
job fails mid-run because the tunnel dropped, just re-run Cell 2 — it kills
the old tunnel and server and opens fresh ones. You'll need to update
`CHAR3D_API_BASE` in the frontend (or console) with the new URL afterward.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `curl: (7) Failed to connect` on health check | Server didn't start — check the `server` process for a traceback, port may already be in use |
| Frontend shows "Submit failed" | `CHAR3D_API_BASE` not set or wrong, or ngrok tunnel expired/changed URL |
| Job stuck on "queued" forever | Worker thread didn't start — check for an exception in the first job's traceback via `GET /jobs/{job_id}` |
| `CORS` errors in browser console | Shouldn't happen (CORS is wide open in `main.py`), but if you tighten `allow_origins` later, make sure your frontend's actual origin is included |
| GPU OOM once real backends are wired in | Confirm `BackendManager` is actually unloading between jobs — check `nvidia-smi` before/after a job completes |
| Job fails at rigging with "Blender executable not found" | `BLENDER_EXECUTABLE` env var not set, or set before the wrong cell/kernel restart — re-run step 2b, then restart the uvicorn cell so it inherits the env |
| Job fails at rigging with "No usable mesh" | The generation stage is still a stub for that backend (mesh.export() not wired in yet) — rigging can't run on a file that was never written; fill in that backend's `generate()` first |
| Rig report shows `used_fallback_leg_split: true` | Normal for dress/saree-style geometry where the lower body doesn't split into two leg clusters — proportional placement was used instead; rig is still valid, just less geometry-informed |
