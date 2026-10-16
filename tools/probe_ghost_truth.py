"""Decisive test: is a GHOST box a real enemy, or a slotless leftover?

The GHOST overlay keys purely off the salmon palette (236,106,100).  If a
live enemy sprite happens to be that colour AND its object slot is at a
different screen position than the rendered sprite (or is not read at
all), the enemy would be swallowed by GHOST.

This probe plays the agent, and whenever a ghost blob is detected it dumps
the full slot table (id/x/y/vx) plus the ghost centre, so we can see
whether any real slot sits under the ghost.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

import numpy as np
import torch
from cynes import NES
from PIL import Image

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, reset_cam
from train import ActorCritic
from visualize import boot, objects, _ghost_blobs, GHOST_RGB

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--max-frames", type=int, default=12000)
ap.add_argument("--max-ghosts", type=int, default=12)
args = ap.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
net.load_state_dict(ck["model"])
net.eval()

nes = NES(ROM)
boot(nes)
reset_cam(nes)

action = 0
prev_lives = nes[R.LIVES]
found = 0
saved = 0


def slot_table(nes):
    rows = []
    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + s]
        x = R.px(R.read_obj_world_x(nes, s))
        y = R.px(R.read_obj_world_y(nes, s))
        rows.append((s, oid, x, y))
    return rows


for i in range(args.max_frames):
    if i % 4 == 0:
        with torch.no_grad():
            action = int(net(obs_tensor(nes))[0].argmax())
    nes.controller = ACTIONS[action][0]
    frame = np.asarray(nes.step(), dtype=np.uint8)

    objs = objects(nes)
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px_x, px_y = (player[2], player[3]) if player else (0, 0)
    # NOTE: deliberately do NOT filter zombies here -- we want the whole
    # truth: does the ghost sit on a slot that `objects()` returned?
    exclude = [(px_x, px_y)] + [(o[2], o[3]) for o in objs]
    ghosts = _ghost_blobs(frame, exclude)

    if nes[R.LIVES] < prev_lives:
        print(f"f{i}: DEATH lives={nes[R.LIVES]}")
        prev_lives = nes[R.LIVES]

    for gx, gy, npx in ghosts:
        found += 1
        if found > args.max_ghosts:
            continue
        # sample the actual colour at the ghost centre
        cx, cy = int(round(gx)), int(round(gy))
        col = frame[min(239, max(0, cy)), min(255, max(0, cx))]
        # nearest slot (any of the 16, including camera slot 15)
        best = None
        for s, oid, x, y in slot_table(nes):
            d = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
            if best is None or d < best[0]:
                best = (d, s, oid, x, y)
        # nearest ENEMY slot specifically
        best_en = None
        for s, oid, x, y in slot_table(nes):
            if oid in R.ENEMY_IDS:
                d = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
                if best_en is None or d < best_en[0]:
                    best_en = (d, s, oid, x, y)
        print(f"\nf{i}: GHOST #{found} at ({gx:.0f},{gy:.0f}) px_n={npx} "
              f"colour={tuple(int(c) for c in col)}")
        print(f"   player=({px_x:.0f},{px_y:.0f}) scroll={nes[R.CAMERA]} "
              f"lives={nes[R.LIVES]} DEATH={nes[R.DEATH]}")
        d, s, oid, x, y = best
        print(f"   nearest slot : s{s} id={oid} @({x:.0f},{y:.0f}) dist={d:.1f}")
        if best_en:
            de, se, oide, xe, ye = best_en
            print(f"   nearest ENEMY: s{se} id={oide} @({xe:.0f},{ye:.0f}) dist={de:.1f}")
        else:
            print("   nearest ENEMY: <none>")
        # full slot table (only non-empty ones)
        nz = [(s, oid, f"{x:.0f}", f"{y:.0f}")
              for s, oid, x, y in slot_table(nes) if oid]
        print(f"   non-empty slots: {nz}")

        if saved < args.max_ghosts:
            from PIL import ImageDraw
            im = Image.fromarray(frame.copy())
            dd = ImageDraw.Draw(im)
            dd.rectangle([gx - 10, gy - 14, gx + 10, gy + 5],
                         outline=(255, 255, 255), width=1)      # ghost
            for s2, oid2, x2, y2 in slot_table(nes):
                if oid2:
                    dd.rectangle([x2 - 8, y2 - 18, x2 + 8, y2 + 4],
                                 outline=(0, 255, 0), width=1)   # every slot
            dd.rectangle([px_x - 8, px_y - 20, px_x + 8, px_y + 4],
                         outline=(255, 0, 255), width=1)         # player
            im.resize((512, 480), Image.NEAREST).save(
                f"outputs/ghosttruth_{saved:02d}.png")
            saved += 1

print(f"\n=== done: {found} ghost boxes over {args.max_frames} frames ===")
nes.controller = 0
del nes
