"""Is $0593 the camera scroll, or the x of a real object in slot 15?

Freeze the player (action 0) while a snail is on screen.  A camera
register must stay put; a walking snail's screen x keeps moving.
Slot 15's id and position are logged alongside the visible snail blob.
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
from visualize import boot, _ghost_blobs

ck = torch.load("runs/ppo/ppo_00092160.pt", map_location="cpu",
                weights_only=False)
net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
net.load_state_dict(ck["model"])
net.eval()

nes = NES(ROM)
boot(nes)
st = nes.save()
reset_cam(nes)

action = 0
done = 0

for i in range(30000):
    if i % 4 == 0:
        with torch.no_grad():
            action = int(net(obs_tensor(nes))[0].argmax())
    nes.controller = ACTIONS[action][0]
    fr = np.asarray(nes.step(), dtype=np.uint8)
    push_img(nes, fr)

    if done >= 2:
        break
    if i < 400:
        if nes[R.LIVES] == 0:
            nes.load(st)
            reset_cam(nes)
        continue

    px = nes[R.OBJ.x_lo] + nes[R.OBJ.x_fr] / 256.0
    blobs = [(gx, gy, n) for gx, gy, n in _ghost_blobs(fr, [(px, nes[R.OBJ.y_lo])])
             if 140 <= gy <= 185 and n >= 100]
    if not blobs:
        if nes[R.LIVES] == 0:
            nes.load(st)
            reset_cam(nes)
        continue

    gx = blobs[0][0]
    print(f"\n--- 冻结玩家 240 帧: 起始 $0593={nes[R.CAMERA]} "
          f"槽15(id={nes[R.OBJ.id+15]}, x_lo={nes[R.OBJ.x_lo+15]}) "
          f"蜗牛块x={gx:.0f} 玩家x={px:.0f} ---")
    for j in range(24):
        nes.controller = 0                     # stand perfectly still
        fr = np.asarray(nes.step(), dtype=np.uint8)
        push_img(nes, fr)
        if j % 4 != 3:
            continue
        px = nes[R.OBJ.x_lo] + nes[R.OBJ.x_fr] / 256.0
        b = [(bx, by, n) for bx, by, n in _ghost_blobs(fr, [(px, nes[R.OBJ.y_lo])])
             if 140 <= by <= 185 and n >= 100]
        bx = b[0][0] if b else -1
        print(f"   +{j+1:2d}: $0593={nes[R.CAMERA]:3d} "
              f"槽15(id={nes[R.OBJ.id+15]:3d}, x_lo={nes[R.OBJ.x_lo+15]:3d}) "
              f"蜗牛块x={bx:6.1f} 玩家x={px:6.1f} lives={nes[R.LIVES]}")
    done += 1

print("\n=== done ===")
nes.controller = 0
del nes
