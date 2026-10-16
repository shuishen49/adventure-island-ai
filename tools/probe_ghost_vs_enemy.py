"""Does a GHOST box ever sit on a REAL enemy?

Plays episodes (auto-restart on game over).  Whenever a live enemy slot
is on screen: census the sprite colours under its box, then run the
GHOST detector and check whether any ghost lands on the enemy.  Also
dumps the FULL slot table (incl. slot 15) at those moments.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tools")

import numpy as np
import torch
from collections import Counter
from cynes import NES
from PIL import Image

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, reset_cam
from train import ActorCritic
from visualize import boot, objects, _ghost_blobs

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--episodes", type=int, default=6)
ap.add_argument("--max-frames", type=int, default=120000)
args = ap.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
net.load_state_dict(ck["model"])
net.eval()

nes = NES(ROM)
boot(nes)
start_state = nes.save()
reset_cam(nes)

enemy_frames = 0
ghost_on_enemy = 0
salmon_censuses = 0
slot_dumps = 0
saved = 0
action = 0

for ep in range(args.episodes):
    if ep:
        nes.load(start_state)
        reset_cam(nes)
    for i in range(args.max_frames // args.episodes):
        if i % 4 == 0:
            with torch.no_grad():
                action = int(net(obs_tensor(nes))[0].argmax())
        nes.controller = ACTIONS[action][0]
        fr = np.asarray(nes.step(), dtype=np.uint8)
        from enjoy import push_img
        push_img(nes, fr)          # keep the visual obs buffer fresh!
        objs = objects(nes)
        player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
        px_x, px_y = (player[2], player[3]) if player else (0, 0)
        ens = [(s, oid, x, y, v) for s, oid, x, y, v in objs
               if oid in R.ENEMY_IDS and 0 <= x < 256 and 0 <= y < 240]
        if not ens:
            if nes[R.LIVES] == 0:
                break
            continue
        enemy_frames += 1
        if nes[R.LIVES] == 0:
            break

        if salmon_censuses < 6:
            s, oid, x, y, v = ens[0]
            reg = fr[max(0, int(y) - 24):int(y) + 4,
                     max(0, int(x) - 10):int(x) + 10].reshape(-1, 3)
            m = ((np.abs(reg[:, 0] - 236) < 26)
                 & (np.abs(reg[:, 1] - 106) < 26)
                 & (np.abs(reg[:, 2] - 100) < 26))
            cols = Counter(map(tuple, reg.astype(int)))
            print(f"e{ep}f{i}: ENEMY s{s} id{oid}@({x:.0f},{y:.0f}) "
                  f"salmon={int(m.sum())}/{len(reg)} "
                  f"colors={cols.most_common(3)}")
            salmon_censuses += 1

        if slot_dumps < 6 or enemy_frames % 400 == 0:
            nz = [(s, oid, f"{x:.0f}", f"{y:.0f}")
                  for s, oid, x, y in
                  [(ss, nes[R.OBJ.id + ss],
                    R.px(R.read_obj_world_x(nes, ss)),
                    R.px(R.read_obj_world_y(nes, ss)))
                   for ss in range(R.N_OBJ_SLOTS)] if oid]
            print(f"e{ep}f{i}: slots={nz}")
            slot_dumps += 1

        exclude = [(px_x, px_y)] + [(o[2], o[3]) for o in objs]
        ghosts = _ghost_blobs(fr, exclude)
        for gx, gy, n in ghosts:
            for s, oid, x, y, v in ens:
                if abs(gx - x) < 20 and abs(gy - y) < 26:
                    ghost_on_enemy += 1
                    if ghost_on_enemy <= 4 and saved < 6:
                        im = Image.fromarray(fr.copy())
                        from PIL import ImageDraw
                        dd = ImageDraw.Draw(im)
                        dd.rectangle([gx - 10, gy - 14, gx + 10, gy + 5],
                                     outline=(255, 255, 255), width=1)
                        dd.rectangle([x - 8, y - 20, x + 8, y + 4],
                                     outline=(0, 120, 255), width=1)
                        im.resize((512, 480), Image.NEAREST).save(
                            f"outputs/ghostonenemy_{saved}.png")
                        saved += 1
                        print(f"  !!! GHOST ({gx:.0f},{gy:.0f}) ON enemy "
                              f"s{s} id{oid}@({x:.0f},{y:.0f})")
                    break

print(f"\n=== enemy frames={enemy_frames}, ghost-on-enemy={ghost_on_enemy} ===")
nes.controller = 0
del nes
