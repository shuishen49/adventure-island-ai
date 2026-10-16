"""Ground-truth test for slot 15: real object slot, or scroll register?

Plays the agent and, for every enemy slot, counts how often its decoded
position lands on a visible ground snail (salmon pixels in the ground
row).  Comparing the "decode" rule against a "screen byte only" rule
shows which one describes the rendered sprite.

Result (2026-10-13, 15k frames, checkpoint ppo_00092160.pt):
    all enemy slots  decode    : 22542 boxes, 60.1% on a snail
                     screenbyte: 30336 boxes, 46.0% on a snail
    slot 15 only     decode    :  5823 boxes, 99.3% on a snail (41 empty)
                     screenbyte:  6665 boxes, 89.0% on a snail
=> slot 15 is a NORMAL object slot that usually holds the ground snail;
   skipping it (as the old "camera slot" rule did) is why snails showed
   up unboxed.  The 3-byte decode is the correct position reading.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

import argparse

import numpy as np
import torch
from cynes import NES

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, push_img, reset_cam
from train import ActorCritic
from visualize import boot

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_00092160.pt")
ap.add_argument("--frames", type=int, default=15000)
args = ap.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
net.load_state_dict(ck["model"])
net.eval()

nes = NES(ROM)
boot(nes)
start_state = nes.save()
reset_cam(nes)

action = 0
stat = {"decode": [0, 0], "screenbyte": [0, 0]}
only15 = {"decode": [0, 0], "screenbyte": [0, 0]}

for i in range(args.frames):
    if i % 4 == 0:
        with torch.no_grad():
            action = int(net(obs_tensor(nes))[0].argmax())
    nes.controller = ACTIONS[action][0]
    frame = np.asarray(nes.step(), dtype=np.uint8)
    push_img(nes, frame)

    reg = frame[140:186].astype(int)
    m = ((np.abs(reg[:, :, 0] - 236) < 26)
         & (np.abs(reg[:, :, 1] - 106) < 26)
         & (np.abs(reg[:, :, 2] - 100) < 26))
    colcnt = m.sum(axis=0)

    def snail_at(x: float) -> bool:
        xi = int(round(x))
        if not (0 <= xi < 256):
            return False
        return colcnt[max(0, xi - 14):xi + 14].sum() >= 30

    for s in range(1, R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + s]
        if oid not in R.ENEMY_IDS:
            continue
        sb = nes[R.OBJ.x_lo + s] + nes[R.OBJ.x_fr + s] / 256.0
        dec = nes[R.OBJ.x_hi + s] * 256 + sb
        ylo = nes[R.OBJ.y_lo + s]
        for name, x in (("decode", dec), ("screenbyte", sb)):
            if 0 <= x < 256 and 0 <= ylo < 240:
                stat[name][0] += 1
                hit = snail_at(x)
                stat[name][1] += hit
                if s == 15:
                    only15[name][0] += 1
                    only15[name][1] += hit

    if nes[R.LIVES] == 0:
        nes.load(start_state)
        reset_cam(nes)

print("all enemy slots:")
for k, (b, h) in stat.items():
    print(f"  {k:11s}: boxes {b:6d}, on a snail {h:6d} "
          f"({100 * h / max(1, b):.1f}%), empty {b - h}")
print("slot 15 only:")
for k, (b, h) in only15.items():
    print(f"  {k:11s}: boxes {b:6d}, on a snail {h:6d} "
          f"({100 * h / max(1, b):.1f}%), empty {b - h}")

nes.controller = 0
del nes
