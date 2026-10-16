"""Render one annotated frame with and without the camera-slot ghost box.

Old behaviour: ``objects()`` scanned all 16 slots, so slot 15 (the camera /
scroll object, id 14) was drawn as a red ENEMY box whenever its garbage
coordinates happened to fall inside the screen.  This script finds such a
frame and saves a side-by-side image: left = old, right = fixed.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")

from cynes import NES

import ram_map as R
import visualize as V
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, push_img, reset_cam
from train import ActorCritic

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo/ppo_00081920.pt"
OUT = "outputs/ghostbox_fix.png"

_fixed_objects = V.objects


def objects_old(nes):
    """The pre-fix scanner: every slot, camera slot included."""
    out = []
    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        if oid == 0:
            continue
        x = R.px(R.read_obj_world_x(nes, s))
        y = R.px(R.read_obj_world_y(nes, s))
        vx = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
        vx = vx - 256 if vx > 127 else vx
        out.append((s, oid, x, y, vx))
    return out


def ghosts(nes):
    """Slot-15 boxes the pre-fix code would (wrongly) have drawn."""
    n = 0
    for s, oid, x, y, _vx in objects_old(nes):
        if s == R.OBJ.CAMERA_SLOT and 0 <= x < 256 and 0 <= y < 240:
            n += 1
    return n


ck = torch.load(CKPT, map_location="cpu", weights_only=False)
net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
net.load_state_dict(ck["model"])
net.eval()

nes = NES(ROM)
V.boot(nes)
reset_cam(nes)

action = 0
total_frames = ghost_frames = 0
shot = None
for i in range(4000):
    if i % 4 == 0:
        with torch.no_grad():
            idx = int(net(obs_tensor(nes))[0].argmax())
        action = ACTIONS[idx][0]
    nes.controller = action
    buf = np.asarray(nes.step(), dtype=np.uint8)
    push_img(nes, buf)
    total_frames += 1
    g = ghosts(nes)
    if g:
        ghost_frames += 1
        if shot is None:
            V.objects = objects_old
            left = V.annotate(buf, nes, "old (with camera slot)")
            V.objects = _fixed_objects
            right = V.annotate(buf, nes, "fixed (camera slot skipped)")
            shot = np.concatenate([left, right], axis=1)

print(f"frames with a spurious slot-15 box: {ghost_frames}/{total_frames} "
      f"({100 * ghost_frames / total_frames:.0f}%)")

if shot is not None:
    img = Image.fromarray(shot).resize(
        (shot.shape[1] * 2, shot.shape[0] * 2), Image.NEAREST)
    img.save(OUT)
    print(f"saved {OUT}  (left = old, right = fixed)")
else:
    print("no ghost frame found in this run (try more frames)")

del nes
