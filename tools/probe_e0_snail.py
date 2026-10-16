"""Snapshot dump: visible snail on screen while E0 (no enemy slot drawn).

Prints every slot's raw bytes (id, x_hi, x_lo, x_fr, y_lo) so we can see
whether the snail hides in a slot whose 3-byte decode falls outside
0..255 (and would therefore be skipped by annotate()).
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

import numpy as np
import torch
from PIL import Image, ImageDraw
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
st = nes.save()
reset_cam(nes)

action = 0
cases = 0
saved = 0

for i in range(25000):
    if i % 4 == 0:
        with torch.no_grad():
            action = int(net(obs_tensor(nes))[0].argmax())
    nes.controller = ACTIONS[action][0]
    fr = np.asarray(nes.step(), dtype=np.uint8)
    push_img(nes, fr)

    objs = objects(nes)
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px, py = (player[2], player[3]) if player else (0, 0)
    ens_onscreen = [o for o in objs if o[1] in R.ENEMY_IDS
                    and 0 <= o[2] < 256 and 0 <= o[3] < 240]
    if ens_onscreen:
        continue                       # already boxed, not our case

    blobs = [(gx, gy, n) for gx, gy, n in _ghost_blobs(fr, [(px, py)])
             if 140 <= gy <= 185 and n >= 120 and abs(gx - px) > 14]
    if not blobs:
        if nes[R.LIVES] == 0:
            nes.load(st)
            reset_cam(nes)
        continue

    cases += 1
    if cases > 6:
        break
    gx, gy, n = blobs[0]
    print(f"\nf{i}: 可见蜗牛块 ({gx:.0f},{gy:.0f}) n={n}  玩家=({px:.0f},{py:.0f}) "
          f"scroll={nes[R.CAMERA]} lives={nes[R.LIVES]}")
    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + s]
        xhi = nes[R.OBJ.x_hi + s]
        xlo = nes[R.OBJ.x_lo + s]
        xfr = nes[R.OBJ.x_fr + s]
        ylo = nes[R.OBJ.y_lo + s]
        if oid == 0 and xlo == 0:
            continue
        full = R.px(R.read_obj_world_x(nes, s))
        sb = xlo + xfr / 256.0
        flag = ""
        if abs(sb - gx) < 18 and abs(ylo - gy) < 24:
            flag = "  <== 屏幕字节对上蜗牛!"
        print(f"   s{s:2d} id={oid:3d} x_hi={xhi:3d} x_lo={xlo:3d} x_fr={xfr:3d} "
              f"y_lo={ylo:3d} | 解码={full:7.1f} 屏幕字节={sb:6.1f}{flag}")
    if saved < 3:
        im = Image.fromarray(fr.copy())
        d = ImageDraw.Draw(im)
        d.rectangle([gx - 11, gy - 16, gx + 11, gy + 5], outline=(255, 255, 255), width=1)
        im.resize((512, 480), Image.NEAREST).save(f"outputs/e0snail_{saved}.png")
        saved += 1

    if nes[R.LIVES] == 0:
        nes.load(st)
        reset_cam(nes)

print("\n=== done ===")
nes.controller = 0
del nes
