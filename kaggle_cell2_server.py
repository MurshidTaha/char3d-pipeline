# =====================================================================
#  CELL 2 — start uvicorn + open ngrok tunnel (robust: retries stale
#  tunnels, handles "endpoint already online" ERR_NGROK_334, waits for
#  real startup instead of racing the health check).
#
#  Re-run this cell any time the tunnel drops or the server needs a
#  restart (e.g. after a fresh CharacterGen source patch, since Python
#  caches already-imported modules for the life of the process).
# =====================================================================
import os, subprocess, time, signal, traceback

REPO_DIR = "/kaggle/working/char3d-pipeline"
PORT = 8000

STEPS_TOTAL = 4
_n = 0
def step(t):
    global _n; _n += 1
    print(f"\n{'='*70}\n[STEP {_n}/{STEPS_TOTAL}] {t}\n{'='*70}")

try:
    step("GPU sanity check")
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
    if not os.path.isdir(REPO_DIR):
        raise RuntimeError(f"{REPO_DIR} not found — run Cell 1 first.")
    print(f"  ✅ repo found at {REPO_DIR}")

    step("Install / reuse portable Blender (CPU-only rigger, no VRAM used)")
    blender_dir = "/kaggle/working/blender-4.2.3-linux-x64"
    blender_bin = f"{blender_dir}/blender"
    if os.path.exists(blender_bin):
        print(f"  ✅ already installed at {blender_bin}, skipping download")
    else:
        subprocess.run(["wget", "-q",
            "https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz",
            "-O", "/kaggle/working/blender.tar.xz"], check=True)
        subprocess.run(["tar", "-xf", "/kaggle/working/blender.tar.xz", "-C", "/kaggle/working"], check=True)
        print(f"  ✅ Blender installed at {blender_bin}")
    ver = subprocess.run([blender_bin, "--version"], capture_output=True, text=True).stdout.splitlines()
    print(f"  {ver[0] if ver else ''}")

    step("Start uvicorn (kills any stale server on the port first)")
    # Find + kill anything already bound to PORT.
    lsof = subprocess.run(["bash", "-c", f"lsof -ti:{PORT}"], capture_output=True, text=True)
    for pid in lsof.stdout.split():
        print(f"  killing stale process on port {PORT}: pid {pid}")
        try:
            os.kill(int(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1)

    server = subprocess.Popen(
        ["python3", "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=REPO_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1,
    )
    print(f"  uvicorn launched (pid {server.pid}), waiting for it to bind...")

    # Wait for real startup (not just the port opening) by reading the
    # process's own stdout for uvicorn's "Application startup complete"
    # line, instead of racing a curl call against a socket that's open
    # but not yet accepting connections.
    startup_ok = False
    lines_seen = []
    start_time = time.time()
    while time.time() - start_time < 90:
        line = server.stdout.readline()
        if line:
            lines_seen.append(line.rstrip())
            print(f"  {line.rstrip()}")
            if "Application startup complete" in line:
                startup_ok = True
                break
        if server.poll() is not None:
            break
        if not line:
            time.sleep(0.5)

    if not startup_ok:
        raise RuntimeError(
            "uvicorn did not report a clean startup within 90s — see the "
            "output above for a traceback (import error, port conflict, etc.)."
        )
    print(f"  ✅ server listening on 127.0.0.1:{PORT}")

    step("Health check + open ngrok tunnel (retries stale/'already online' tunnels)")
    health = subprocess.run(
        ["curl", "-s", "--max-time", "10", f"http://localhost:{PORT}/health"],
        capture_output=True, text=True
    )
    print(f"  /health -> {health.stdout.strip()!r}")
    if '"ok"' not in health.stdout:
        raise RuntimeError(f"/health didn't return ok: {health.stdout!r}")

    from pyngrok import ngrok, conf

    # Kill the LOCAL agent, then also explicitly disconnect any tunnels
    # ngrok's cloud side still thinks are online (this is what causes
    # ERR_NGROK_334 "endpoint already online" on a fresh ngrok.connect()
    # right after a kernel/session restart).
    ngrok.kill()
    time.sleep(2)
    try:
        for t in ngrok.get_tunnels():
            ngrok.disconnect(t.public_url)
    except Exception:
        pass  # no local agent running yet — fine

    public_url = None
    last_err = None
    for attempt in range(6):
        try:
            public_url = ngrok.connect(addr=f"127.0.0.1:{PORT}", proto="http")
            break
        except Exception as e:
            last_err = e
            print(f"  tunnel attempt {attempt + 1}/6 failed: {e}")
            time.sleep(5)

    if public_url is None:
        raise RuntimeError(
            f"Could not open ngrok tunnel after 6 attempts. Last error: {last_err}\n"
            f"If this persists, the endpoint may be stuck online on ngrok's "
            f"side — visit https://dashboard.ngrok.com/agents and manually "
            f"stop it, then re-run this cell."
        )

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
