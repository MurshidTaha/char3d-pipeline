# Fresh Kaggle setup — quick start

This package already has every fix discovered while debugging Aryan's
first successful run baked in (SD2.1 mirror, peft pin, dataclass patches,
weights_only patch, robust auto-rig GLB export, robust ngrok reconnect,
live progress/logs, embedded 3D viewer, auto-save frontend).

## Steps

1. **Upload this zip as a Kaggle Dataset input** on a fresh notebook:
   sidebar → *Add Input* → *Upload* → select this zip.
   (This replaces the old `git clone` of char3d-pipeline itself — Cell 1
   now unpacks your own updated code instead of pulling from GitHub, so
   you never have to re-discover these fixes on a fresh clone again.
   TripoSR and CharacterGen, the two third-party repos, are still cloned
   from GitHub as before, then auto-patched by Cell 1.)

2. **Cell 1** — paste `kaggle_cell1_setup.py`'s contents into a fresh code
   cell and run it. Installs everything + auto-patches CharacterGen.

3. **Cell 2** — paste `kaggle_cell2_server.py`'s contents into the next
   cell and run it. Starts the server, opens the ngrok tunnel, and prints
   your `API base URL`. Re-run this cell any time the tunnel drops or you
   need to restart the server (e.g. after editing any source file).

4. **Frontend** — open `frontend/index.html` in your browser (or host it
   however you like). If the printed API base URL differs from the one
   already hardcoded at the top of the `<script type="module">` block,
   either edit that line or set `window.CHAR3D_API_BASE = "..."` in the
   browser console before the page's script runs.

5. Submit a character. You'll now see: a live percentage progress bar, a
   scrolling log panel (no more checking Kaggle logs manually), inline
   error messages if something fails, and — once complete — an embedded
   3D viewer you can rotate/zoom right on the page.

6. **Auto-save without manual downloading**: check "auto-save finished
   packages to a local folder", click "choose auto-save folder" once
   (Chrome/Edge only — this uses the File System Access API), and pick
   e.g. `E:\youtube\char3d-pipeline\outputs\`. Every job's `.zip` after
   that saves there automatically with no click needed. This permission
   lasts for the browser session; you'll pick the folder again next time
   you reload the page (browser security — a site can't silently keep
   write access to your filesystem across page loads without you
   re-granting it).

## If something still breaks

Paste the exact error/traceback back — the goal of these Cell 1/2 scripts
is that a fresh Kaggle instance shouldn't need any of the manual patching
we did the first time through, but Kaggle's base image or the third-party
repos can still shift under you over time.
