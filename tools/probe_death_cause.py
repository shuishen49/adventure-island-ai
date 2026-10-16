"""Death causality: does the thing that kills the player have a slot?

Every time a life is lost we look at the last frame:
  * nearest on-screen ENEMY slot (id in ENEMY_IDS) to the player
  * nearest salmon ground blob to the player
If deaths happen with a blob on top of the player but NO slot nearby,
then the slotless snails are REAL killers we fail to track.
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
prev_lives = nes[R.LIVES]
deaths = 0
kind_slot = 0        # killed with an enemy slot right there
kind_blob_only = 0  # killed with a blob but no slot
kind_neither = 0    # pit / stamina / unknown
last = None         # (px, py, nearest_slot_dist, nearest_blob_dist)

for i in range(30000):
    if i % 4 == 0:
        with torch.no_grad():
            action = int(net(obs_tensor(nes))[0].argmax())
    nes.controller = ACTIONS[action][0]
    frame = np.asarray(nes.step(), dtype=np.uint8)
    push_img(nes, frame)

    objs = objects(nes)
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px, py = (player[2], player[3]) if player else (0, 0)

    ens = [(s, oid, x, y) for s, oid, x, y, v in objs
           if oid in R.ENEMY_IDS and 0 <= x < 256 and 0 <= y < 240]
    ds = min((max(abs(x - px), abs(y - py)) for s, oid, x, y in ens),
             default=999)
    # cheap: salmon pixels in a small window around the player's feet
    x0, x1 = max(0, int(px) - 22), min(256, int(px) + 22)
    y0, y1 = max(0, int(py) - 26), min(240, int(py) + 10)
    reg = frame[y0:y1, x0:x1].astype(int)
    m = ((np.abs(reg[:, :, 0] - 236) < 26)
         & (np.abs(reg[:, :, 1] - 106) < 26)
         & (np.abs(reg[:, :, 2] - 100) < 26))
    npx = int(m.sum())
    db = 0 if npx >= 25 else 999
    last = (px, py, ds, db)

    if nes[R.LIVES] < prev_lives:
        deaths += 1
        px, py, ds, db = last
        if ds <= 18:
            kind_slot += 1
            tag = "SLOT"
        elif db <= 18:
            kind_blob_only += 1
            tag = "BLOB-ONLY!!"
        else:
            kind_neither += 1
            tag = "pit/other"
        print(f"f{i}: 死亡 #{deaths} {tag} | player=({px:.0f},{py:.0f}) "
              f"最近敌槽距离={ds} 最近蜗牛块距离={db}")
        prev_lives = nes[R.LIVES]

    if nes[R.LIVES] == 0:
        nes.load(start_state)
        reset_cam(nes)
        prev_lives = nes[R.LIVES]

print(f"\n=== 死亡 {deaths} 次: 有槽敌人={kind_slot}, "
      f"只有无槽蜗牛={kind_blob_only}, 坑/其他={kind_neither} ===")
nes.controller = 0
del nes
