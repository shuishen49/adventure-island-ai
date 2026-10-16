"""Walk into the visible 'slotless' snail -- does it kill the player?

If touching it kills -> it is a REAL enemy (our slot reading misses it).
If the player passes through -> leftover image, GHOST label was right.
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
tests = 0
prev_lives = nes[R.LIVES]


def ground_blobs(fr, px, py):
    """Salmon blobs near ground level, away from the player."""
    blobs = _ghost_blobs(fr, [(px, py)])
    return [(gx, gy, n) for gx, gy, n in blobs
            if 120 <= gy <= 185 and abs(gx - px) > 14]


for i in range(60000):
    objs = objects(nes)
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px, py = (player[2], player[3]) if player else (0, 0)

    if i % 4 == 0:
        with torch.no_grad():
            action = int(net(obs_tensor(nes))[0].argmax())

    nes.controller = ACTIONS[action][0]
    frame = np.asarray(nes.step(), dtype=np.uint8)
    push_img(nes, frame)

    # after stepping, decide whether to charge on the NEXT decision point
    if i % 4 == 3 and tests < 4:
        objs = objects(nes)
        player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
        px, py = (player[2], player[3]) if player else (0, 0)
        gb = ground_blobs(frame, px, py)
        ens = [(s, oid, x, y) for s, oid, x, y, v in objects(nes)
               if oid in R.ENEMY_IDS and 0 <= x < 256]
        if gb:
            gx, gy, n = min(gb, key=lambda b: abs(b[0] - px))
            if abs(gx - px) < 46 and nes[R.LIVES] == prev_lives:
                print(f"\nf{i}: 冲向鲑红块 ({gx:.0f},{gy:.0f}) n={n} | "
                      f"玩家=({px:.0f},{py:.0f}) 敌槽={ens}")
                # walk right into it for up to 80 frames
                touched = False
                for j in range(80):
                    nes.controller = ACTIONS[1][0] if gx > px else ACTIONS[2][0]
                    frame = np.asarray(nes.step(), dtype=np.uint8)
                    push_img(nes, frame)
                    objs = objects(nes)
                    player = next((o for o in objs
                                   if o[0] == R.PLAYER_SLOT), None)
                    px, py = (player[2], player[3]) if player else (px, py)
                    if nes[R.LIVES] < prev_lives:
                        print(f"  f{i+j}: 死亡! 此时玩家=({px:.0f},{py:.0f}) "
                              f"敌槽={[(s, o, f'{x:.0f}', f'{y:.0f}') for s, o, x, y, v in objects(nes) if o in R.ENEMY_IDS]}")
                        touched = True
                        tests += 1
                        prev_lives = nes[R.LIVES]
                        break
                    if abs(px - gx) < 6:
                        print(f"  f{i+j}: 已到达块位置 ({px:.0f} vs {gx:.0f}) "
                              f"未死，敌槽={[(s, o, f'{x:.0f}', f'{y:.0f}') for s, o, x, y, v in objects(nes) if o in R.ENEMY_IDS]}")
                        touched = True
                        tests += 1
                        break
                if not touched:
                    print("  未接触到（跳过了/块消失）")
                    tests += 1

    if nes[R.LIVES] < prev_lives:
        prev_lives = nes[R.LIVES]
    if nes[R.LIVES] == 0:
        nes.load(start_state)
        reset_cam(nes)
        prev_lives = nes[R.LIVES]

print(f"\n=== 冲撞测试完成: {tests} 次 ===")
nes.controller = 0
del nes
