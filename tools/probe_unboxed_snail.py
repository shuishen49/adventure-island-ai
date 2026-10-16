"""Why is a visible snail not boxed?

Hypothesis: object screen position is decoded from 3 bytes (x_hi/x_lo/x_fr),
so a dirty x_hi byte pushes the coordinate outside 0..255 and the box is
skipped -- even though the sprite is on screen.  The screen byte alone
(x_lo + x_fr/256) is enough for drawing.

For every visible ground snail blob we look for a slot whose SCREEN byte
(x_lo, y_lo) sits under it, and report what the 3-byte decode gave.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

import numpy as np
import torch
from cynes import NES

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, push_img, reset_cam
from train import ActorCritic
from visualize import boot, objects, _ghost_blobs

ck = torch.load("runs/ppo/ppo_00092160.pt", map_location="cpu",
                weights_only=False)
net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
net.load_state_dict(ck["model"])
net.eval()

nes = NES(ROM)
boot(nes)
start_state = nes.save()
reset_cam(nes)

action = 0
matched = 0          # blob explained by a slot (any id)
unmatched = 0        # blob with no slot under it
misdecoded = 0       # slot under blob but 3-byte decode is off-screen
examples = 0

for i in range(40000):
    if i % 4 == 0:
        with torch.no_grad():
            action = int(net(obs_tensor(nes))[0].argmax())
    nes.controller = ACTIONS[action][0]
    frame = np.asarray(nes.step(), dtype=np.uint8)
    push_img(nes, frame)

    objs = objects(nes)
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px, py = (player[2], player[3]) if player else (0, 0)
    blobs = [(gx, gy, n) for gx, gy, n in _ghost_blobs(frame, [(px, py)])
             if 140 <= gy <= 185 and n >= 60 and abs(gx - px) > 12]
    if not blobs:
        if nes[R.LIVES] == 0:
            nes.load(start_state)
            reset_cam(nes)
        continue

    # every non-empty slot: screen byte vs full decode
    rows = []
    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + s]
        if oid == 0:
            continue
        x_lo = nes[R.OBJ.x_lo + s]
        y_lo = nes[R.OBJ.y_lo + s]
        x_hi = nes[R.OBJ.x_hi + s]
        full = R.px(R.read_obj_world_x(nes, s))
        rows.append((s, oid, x_lo, y_lo, x_hi, full))

    for gx, gy, n in blobs:
        hit = None
        for s, oid, x_lo, y_lo, x_hi, full in rows:
            if abs(x_lo - gx) < 16 and abs(y_lo - gy) < 22:
                hit = (s, oid, x_lo, y_lo, x_hi, full)
                break
        if hit is None:
            unmatched += 1
            if examples < 6 and unmatched % 50 == 1:
                print(f"f{i}: blob({gx:.0f},{gy:.0f}) n={n} -> NO slot under it; "
                      f"slots={[(r[0], r[1], r[2], r[3], r[4], round(r[5])) for r in rows]}")
        else:
            matched += 1
            s, oid, x_lo, y_lo, x_hi, full = hit
            if not (0 <= full < 256):
                misdecoded += 1
                if examples < 6:
                    print(f"f{i}: blob({gx:.0f},{gy:.0f}) = slot s{s} id{oid} "
                          f"screenbyte=({x_lo},{y_lo}) BUT decode(full)={full:.1f} "
                          f"x_hi={x_hi} -> box skipped! (enemy={oid in R.ENEMY_IDS})")
                    examples += 1
            elif examples < 3:
                print(f"f{i}: blob({gx:.0f},{gy:.0f}) = slot s{s} id{oid} "
                      f"decode={full:.1f} ok")

    if nes[R.LIVES] == 0:
        nes.load(start_state)
        reset_cam(nes)

print(f"\n=== blobs explained by a slot: {matched}  "
      f"(of which misdecoded/off-screen: {misdecoded})  "
      f"| blobs with NO slot: {unmatched} ===")
nes.controller = 0
del nes
