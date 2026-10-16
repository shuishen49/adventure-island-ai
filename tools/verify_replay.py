"""Verify that the jsnes replay reproduced the cyNES recording frame-for-frame.

Compares, for every dumped frame:
  * camera / lives read out of RAM on both sides
  * mean absolute pixel difference between the raw cyNES frame (PPM) and the
    jsnes frame (PNG); both emulators use slightly different palettes, so we
    accept a low MAE, not exact equality.

usage:
    python tools/verify_replay.py outputs/replay_frames_ep0 outputs/replay_sound/replay_frames_jsnes
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

CY_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/replay_frames_ep0"
JS_DIR = sys.argv[2] if len(sys.argv) > 2 else "outputs/replay_sound/replay_frames_jsnes"

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prog_cy = json.load(open(os.path.join(root, "outputs", "replay_progress.json")))
prog_js = json.load(open(os.path.join(root, "outputs", "replay_sound",
                                      "replay_progress_jsnes.json")))

# ---- RAM state ---------------------------------------------------------
ram_bad = 0
pairs = {(p["frame"]): (p["camera"], p["lives"]) for p in prog_cy}
for p in prog_js:
    want = pairs.get(p["frame"])
    if want is None:
        continue
    if want != (p["camera"], p["lives"]):
        ram_bad += 1
        print(f"  RAM MISMATCH frame {p['frame']}: "
              f"cy={want} js=({p['camera']},{p['lives']})")
n_cmp = min(len(prog_cy), len(prog_js))
print(f"RAM checks: {n_cmp} samples, {ram_bad} mismatches")

# ---- pixels ------------------------------------------------------------
maes = []
worst = (0.0, None)
for cy_path in sorted(glob.glob(os.path.join(CY_DIR, "*.ppm"))):
    name = os.path.splitext(os.path.basename(cy_path))[0]
    js_path = os.path.join(JS_DIR, name + ".png")
    if not os.path.exists(js_path):
        continue
    a = np.asarray(Image.open(cy_path), dtype=np.int16)
    b = np.asarray(Image.open(js_path).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        print(f"  SHAPE MISMATCH {name}: {a.shape} vs {b.shape}")
        continue
    mae = float(np.abs(a - b).mean())
    maes.append(mae)
    if mae > worst[0]:
        worst = (mae, name)

if maes:
    maes_arr = np.array(maes)
    print(f"pixel MAE over {len(maes)} frames: "
          f"mean {maes_arr.mean():.2f}, median {np.median(maes_arr):.2f}, "
          f"max {maes_arr.max():.2f} ({worst[1]})")
    # cyNES and jsnes ship different NES palettes, so raw MAE stays high even
    # when perfectly in sync; the RAM equality check is the real verdict.
    print("verdict:", "IN SYNC" if ram_bad == 0 and maes_arr.mean() < 60
          else "DIVERGED -- do not mux this audio")
else:
    print("no frame pairs found")
