# Setup Guide — Character-to-3D Tool

Step-by-step, from a fresh Kaggle notebook to your first downloaded package.
Assumes the same Kaggle + ngrok + Windows-client pattern you're already using
for Phase 1/6.

---

## 0. Prerequisites

- Kaggle account with GPU quota (T4 x1, 16GB)
- ngrok account (free tier is fine) + authtoken
- The `char3d_tool.zip` from this conversation, unzipped somewhere accessible
  to upload into a Kaggle notebook (or push it to a GitHub repo and clone it
  in-notebook — cleaner if you're iterating a lot)

---

## 1. Get the code into Kaggle

**Option A — upload as a Kaggle Dataset (simplest for one-off use):**
1. Kaggle → Datasets → New Dataset → upload `char3d_tool.zip`
2. In your notebook: Add Data → your dataset
3. Unzip it into the working directory:
   ```python
   import zipfile
   zipfile.ZipFile('/kaggle/input/char3d-tool/char3d_tool.zip').extractall('/kaggle/working/')
   %cd /kaggle/working/char3d_tool
   ```

**Option B — push to GitHub and clone (better if you'll keep editing it):**
```python
!git clone https://github.com/<your-username>/char3d_tool.git
%cd char3d_tool
```

---

## 2. Install dependencies

Start with the base requirements — this gets the API/queue/routing layer running
with all four backends still stubbed:

```python
!pip install -r backend/requirements.txt
```

Once you start filling in real backend calls (see README's "Fill-in order"),
add packages per backend as you go rather than installing everything up front:

```python
# only when you actually wire in real inference —
# exact packages depend on which repo commit you pin for each backend
!pip install torch torchvision trimesh huggingface_hub diffusers transformers
```

Don't install torch/diffusers/etc. before you need them — Kaggle sessions have
a time budget and unused heavy installs just burn it.

---

## 2b. Installing Blender on Kaggle (for the self-hosted rigger)

Rigging no longer depends on Mixamo — it runs on Blender headless instead
(see `backend/rigging.py` for why). Blender isn't a pip package, so grab the
portable Linux tarball once per session:

```python
import subprocess, os

BLENDER_VERSION = "4.2.3"  # check https://download.blender.org/release/ for current LTS
tarball = f"blender-{BLENDER_VERSION}-linux-x64.tar.xz"
subprocess.run(["wget", "-q", f"https://download.blender.org/release/Blender{BLENDER_VERSION[:3]}/{tarball}"], cwd="/kaggle/working", check=True)
subprocess.run(["tar", "-xf", tarball], cwd="/kaggle/working", check=True)

os.environ["BLENDER_EXECUTABLE"] = f"/kaggle/working/blender-{BLENDER_VERSION}-linux-x64/blender"
```

No `sudo`/apt install needed — the tarball is self-contained and runs from
wherever you extract it. `rigging.py` reads `BLENDER_EXECUTABLE` from the
environment, so set it before starting the FastAPI server (uvicorn inherits
the notebook's env when launched via `subprocess.Popen` in the same cell/session).

Sanity check:
```python
!$BLENDER_EXECUTABLE --version
```

This step is CPU-only — it doesn't touch the T4, so it's safe to run
alongside Phase 5 or any of the 3D generation backends without VRAM contention.

---

## 3. Enable GPU on the notebook

Notebook settings (right panel) → Accelerator → **GPU T4 x1**. Restart the
kernel if you changed this after already running cells.

Sanity check:
```python
!nvidia-smi
```

---

## 4. Set up ngrok

```python
!pip install pyngrok -q
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")  # from ngrok.com dashboard
```

Set this as a **Kaggle Secret** rather than pasting it in plaintext if you're
going to share/publish the notebook:
```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("NGROK_AUTHTOKEN")
ngrok.set_auth_token(token)
```

---

## 5. Start the FastAPI server

Run this in a notebook cell (it needs to run in the background, so use `nohup`
+ `&`, same pattern as your other phases):

```python
import subprocess
server = subprocess.Popen(
    ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd="/kaggle/working/char3d_tool",
)
```

Give it a few seconds to boot, then check:
```python
!curl http://localhost:8000/health
# expect: {"status":"ok"}
```

---

## 6. Open the ngrok tunnel

```python
public_url = ngrok.connect(8000)
print(public_url)
```

Copy the printed URL (looks like `https://xxxx-xx-xx-xx-xx.ngrok-free.app`) —
this is what your Windows client / frontend will hit.

---

## 7. Point the frontend at your tunnel

On your **Windows machine** (`E:\youtube`), open `frontend/index.html` — but
first tell it which API to talk to. Easiest way: add a line right before the
`<script>` tag's closing, or just before it loads, in `index.html`:

```html
<script>window.CHAR3D_API_BASE = "https://xxxx-xx-xx-xx-xx.ngrok-free.app";</script>
```

Add that line just above the existing `<script>` block near the bottom of
`frontend/index.html`, then open the file in a browser (double-click works,
no local server needed).

---

## 8. Verify routing before running real jobs

Check the routing table is loaded correctly:
```python
!curl http://localhost:8000/config/routing
```

This should return the JSON routing table (human/anime → charactergen, etc.).
If you ever want to re-tune a route without redeploying, `PUT` a new table to
this same endpoint.

---

## 9. Run your first job (with stubs)

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
headless → Mixamo automation → the other three backends → Real-ESRGAN).

---

## 10. Keeping the tunnel alive across a long session

ngrok free-tier tunnels can drop on inactivity or Kaggle session limits. If a
job fails mid-run because the tunnel dropped:
```python
ngrok.kill()
public_url = ngrok.connect(8000)
print(public_url)
```
You'll need to update `CHAR3D_API_BASE` in the frontend again after this.

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
