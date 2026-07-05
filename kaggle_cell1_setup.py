# =====================================================================
#  CELL 1 — repo + all deps + TripoSR + CharacterGen + auto-patches
#  Fresh notebook: upload the char3d-pipeline zip as a Kaggle Dataset
#  input first (Add Input -> Upload -> this zip), then paste this in
#  ONE cell and run it.
#
#  This consolidates EVERY environment/compat fix discovered while
#  debugging this pipeline on 2026-07-05:
#    - stabilityai/stable-diffusion-2-1 was deprecated on HF -> mirror
#    - peft must be pinned (0.7.1, --no-deps) -- diffusers imports it
#      unconditionally, and newer peft needs a newer accelerate than
#      the Nov-2023-era stack CharacterGen pins
#    - CharacterGen's lrm/**/*.py has Python-3.9-era dataclass fields
#      with mutable defaults (e.g. `loss: X = X()`), which Python 3.12
#      rejects -- rewritten to use default_factory
#    - lrm/utils/misc.py's torch.load() needs weights_only=False for
#      PyTorch 2.6+ (checkpoint stores an omegaconf.ListConfig)
# =====================================================================
import os, sys, subprocess, traceback, glob, re

REPO_DIR      = "/kaggle/working/char3d-pipeline"
TRIPOSR_URL   = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_DIR   = f"{REPO_DIR}/TripoSR"
CHARGEN_URL   = "https://github.com/zjp-shadow/CharacterGen.git"
CHARGEN_DIR   = f"{REPO_DIR}/CharacterGen"
NGROK_TOKEN   = "3D5SVEQIWSudoIBUYqrHx7l8yVU_7k6ZA6rDCQeaX1c6hgEzL"

# Path to the char3d-pipeline zip you uploaded as a Kaggle Dataset input.
# Adjust the dataset slug if you named it something else on upload.
UPLOADED_ZIP_CANDIDATES = glob.glob("/kaggle/input/*/char3d-pipeline*.zip") + \
                           glob.glob("/kaggle/input/*/*.zip")

STEPS_TOTAL = 10
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
    step("Unpack YOUR char3d-pipeline (from uploaded dataset, not GitHub) —"
         " guarantees the Kaggle instance always runs your latest local fixes")
    if not UPLOADED_ZIP_CANDIDATES:
        raise RuntimeError(
            "No uploaded zip found under /kaggle/input/. Add this zip as a "
            "Kaggle Dataset input first (Notebook sidebar -> Add Input -> "
            "Upload), then re-run this cell."
        )
    zip_path = UPLOADED_ZIP_CANDIDATES[0]
    print(f"  using uploaded zip: {zip_path}")
    os.makedirs(REPO_DIR, exist_ok=True)
    run(["unzip", "-oq", zip_path, "-d", REPO_DIR])
    print(f"  ✅ repo ready at {REPO_DIR}")

    step("Clone TripoSR (third-party, lives inside char3d-pipeline)")
    if os.path.exists(f"{TRIPOSR_DIR}/tsr/system.py"):
        print(f"  already cloned at {TRIPOSR_DIR}, skipping")
    else:
        run(["git", "clone", TRIPOSR_URL, TRIPOSR_DIR])
    print(f"  ✅ TripoSR ready at {TRIPOSR_DIR}")

    step("Clone CharacterGen (third-party, lives inside char3d-pipeline)")
    if os.path.exists(f"{CHARGEN_DIR}/webui.py"):
        print(f"  already cloned at {CHARGEN_DIR}, skipping")
    else:
        run(["git", "clone", CHARGEN_URL, CHARGEN_DIR])
    print(f"  ✅ CharacterGen ready at {CHARGEN_DIR}")

    step("Install pyngrok + authenticate")
    run(["pip", "install", "-q", "pyngrok"])
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_TOKEN)
    print("  ✅ ngrok authenticated")

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

    step("Install CharacterGen's own requirements.txt (era-matched pins)")
    # See original Cell 1 comments for the full rationale on each of these —
    # short version: CharacterGen's Nov-2023 stack needs transformers/
    # accelerate/tokenizers/safetensors/huggingface_hub pinned together,
    # rm_anime_bg installed from GitHub (PyPI metadata is stale),
    # nvdiffrast built with --no-build-isolation, and peft pinned to an
    # era that predates accelerate's clear_device_cache (needed by newer
    # peft but not present in the old accelerate CharacterGen wants).
    chargen_reqs = f"{CHARGEN_DIR}/requirements.txt"
    if os.path.exists(chargen_reqs):
        with open(chargen_reqs) as f:
            lines = f.readlines()
        filtered = [l for l in lines
                    if "nvdiffrast" not in l and "rm_anime_bg" not in l]
        filtered_reqs = f"{CHARGEN_DIR}/requirements_filtered.txt"
        with open(filtered_reqs, "w") as f:
            f.writelines(filtered)

        run(["pip", "install", "-q", "--break-system-packages", "-r", filtered_reqs])
        print("  ✅ CharacterGen requirements installed (excluding rm_anime_bg, nvdiffrast)")

        run(["pip", "install", "-q", "--break-system-packages", "--no-build-isolation",
             "git+https://github.com/NVlabs/nvdiffrast"])
        print("  ✅ nvdiffrast built against Kaggle's existing torch")

        run(["pip", "install", "-q", "--break-system-packages", "--force-reinstall", "--no-deps",
             "transformers==4.35.2", "accelerate==0.24.1", "huggingface_hub==0.19.4",
             "tokenizers==0.15.0", "safetensors==0.4.1"])
        print("  ✅ transformers/accelerate/huggingface_hub/tokenizers/safetensors pinned")

        # peft: diffusers' _unwrap_model() does an UNCONDITIONAL
        # `from peft import PeftModel` (not try/except-guarded), so peft
        # can't just be removed like a normal optional dep. But newer peft
        # needs accelerate.utils.memory.clear_device_cache, which doesn't
        # exist in the pinned accelerate==0.24.1 above. 0.7.1 is the
        # confirmed-working era-matched version. --no-deps so it can't
        # drag in a newer accelerate/transformers and re-break the pins.
        run(["pip", "install", "-q", "--break-system-packages", "--no-deps", "peft==0.7.1"])
        print("  ✅ peft==0.7.1 installed (--no-deps, era-matched to avoid clear_device_cache ImportError)")

        run(["pip", "install", "-q", "--break-system-packages", "--no-deps",
             "rm_anime_bg[cpu] @ git+https://github.com/shirayu/rm_anime_bg.git"])
        print("  ✅ rm_anime_bg installed from GitHub source, --no-deps")
    else:
        print(f"  ⚠️  no requirements.txt found at {chargen_reqs}, skipping")

    step("Let pip's resolver reconcile numpy, then clean-reinstall ABI-bound packages")
    run(["pip", "install", "-q", "--break-system-packages", "--upgrade", "rembg[cpu]"])
    result = subprocess.run(["pip", "show", "numpy"], capture_output=True, text=True)
    numpy_version = next(
        (line.split(":", 1)[1].strip() for line in result.stdout.splitlines()
         if line.startswith("Version:")), "unknown")
    print(f"  resolver settled on numpy {numpy_version}")

    abi_bound = ["numba", "trimesh", "scipy", "pymatting",
                 "opencv-python-headless", "opencv-python",
                 "scikit-image", "onnxruntime", "shapely"]
    for pkg in abi_bound:
        check = subprocess.run(["pip", "show", pkg], capture_output=True, text=True)
        if check.returncode == 0:
            run(["pip", "uninstall", "-y", "-q", pkg])
    run(["pip", "install", "-q", "--break-system-packages"] + abi_bound)
    print(f"  ✅ numpy {numpy_version} + ABI-bound packages reconciled")

    os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/local/cuda/bin"
    run(["pip", "install", "-q", "--break-system-packages", "ninja"])
    run(["pip", "install", "--break-system-packages", "--force-reinstall", "--no-deps",
         "git+https://github.com/tatsy/torchmcubes.git"])
    print(f"  ✅ torchmcubes rebuilt against numpy {numpy_version}")

    step("Auto-patch CharacterGen source for Python 3.12 / modern PyTorch")
    # 1) SD 2.1 was deprecated by Stability AI on HF (Dec 2025, ahead of EU
    #    AI Act compliance) -- swap to the live community mirror. Same
    #    weights/layout, drop-in replacement.
    infer_yaml = f"{CHARGEN_DIR}/2D_Stage/configs/infer.yaml"
    if os.path.exists(infer_yaml):
        with open(infer_yaml) as f:
            content = f.read()
        old_id, new_id = "stabilityai/stable-diffusion-2-1", "sd2-community/stable-diffusion-2-1"
        if old_id in content:
            with open(infer_yaml, "w") as f:
                f.write(content.replace(old_id, new_id))
            print(f"  ✅ patched {infer_yaml}: {old_id} -> {new_id}")
        else:
            print(f"  (already patched or id not found in {infer_yaml})")
    else:
        print(f"  ⚠️  {infer_yaml} not found, skipping SD2.1 mirror patch")

    # 2) Python 3.12 tightened dataclasses: mutable defaults (including
    #    another dataclass instance) now require default_factory. Scan
    #    every file under lrm/ for the `field: Cls = Cls()` pattern and
    #    rewrite it. (CharacterGen's README targets Python 3.9, which was
    #    lenient here.)
    lrm_dir = f"{CHARGEN_DIR}/3D_Stage/lrm"
    pattern = re.compile(r'(\w+):\s*(\w+)\s*=\s*\2\(\)')
    patched_dataclass_files = []
    for path in glob.glob(f"{lrm_dir}/**/*.py", recursive=True):
        with open(path) as f:
            src = f.read()
        matches = pattern.findall(src)
        if not matches:
            continue
        new_src = pattern.sub(
            lambda m: f"{m.group(1)}: {m.group(2)} = field(default_factory={m.group(2)})",
            src
        )
        if "from dataclasses import" in new_src:
            first_line = new_src.split("from dataclasses import", 1)[1].split("\n", 1)[0]
            if "field" not in first_line:
                new_src = new_src.replace(
                    f"from dataclasses import{first_line}",
                    f"from dataclasses import{first_line}, field", 1
                )
        elif "field(default_factory" in new_src:
            new_src = "from dataclasses import field\n" + new_src
        with open(path, "w") as f:
            f.write(new_src)
        patched_dataclass_files.append((path, matches))
    for path, matches in patched_dataclass_files:
        print(f"  ✅ {path}: {matches}")
    if not patched_dataclass_files:
        print("  (no mutable-default dataclass fields found — already patched?)")

    # 3) PyTorch 2.6 changed torch.load()'s default from weights_only=False
    #    to True. CharacterGen's checkpoint carries an omegaconf.ListConfig
    #    alongside tensors, which the strict loader rejects. This is your
    #    own downloaded, trusted checkpoint, so weights_only=False is safe.
    misc_path = f"{lrm_dir}/utils/misc.py"
    if os.path.exists(misc_path):
        with open(misc_path) as f:
            content = f.read()
        old_call = "ckpt = torch.load(path, map_location=map_location)"
        new_call = "ckpt = torch.load(path, map_location=map_location, weights_only=False)"
        if old_call in content:
            with open(misc_path, "w") as f:
                f.write(content.replace(old_call, new_call))
            print(f"  ✅ patched {misc_path} to use weights_only=False")
        else:
            print(f"  (already patched or call signature changed in {misc_path})")
    else:
        print(f"  ⚠️  {misc_path} not found, skipping weights_only patch")

    step("Final verification — import everything CELL 2 / the pipeline will need")
    # Runs in a FRESH subprocess deliberately — this notebook's own kernel
    # has already imported torch/numpy etc. by this point, and re-importing
    # a package that was reinstalled on disk mid-session in the SAME
    # process can hit "cannot load module more than once per process".
    checks = {
        "torch": "torch", "torchmcubes": "torchmcubes", "rembg": "rembg",
        "trimesh": "trimesh", "PIL": "PIL", "fastapi": "fastapi", "uvicorn": "uvicorn",
        "peft": "peft",
    }
    verify_lines = ["import sys, traceback"]
    for label, mod in checks.items():
        verify_lines.append(f"""
try:
    __import__("{mod}")
    print("OK:{label}")
except Exception as e:
    print(f"FAIL:{label}:{{type(e).__name__}}: {{e}}")
""")
    verify_script = "\n".join(verify_lines)

    result = subprocess.run([sys.executable, "-c", verify_script],
                             capture_output=True, text=True)
    all_ok = True
    for line in result.stdout.strip().splitlines():
        if line.startswith("OK:"):
            print(f"  ✅ {line[3:]}")
        elif line.startswith("FAIL:"):
            all_ok = False
            print(f"  ❌ {line[5:]}")
    if result.stderr.strip():
        print(result.stderr[-1500:])

    print(f"""
  {"✅ CELL 1 COMPLETE — everything verified (in a clean process), run CELL 2 next." if all_ok else "⚠️  CELL 1 finished but some imports failed above — paste this output back for another look."}
""")

except Exception as e:
    print(f"\n{'!'*70}\n❌ CELL 1 FAILED AT: [STEP {_n}/{STEPS_TOTAL}]\n{'!'*70}")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()
    raise
