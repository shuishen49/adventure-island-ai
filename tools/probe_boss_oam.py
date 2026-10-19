"""Is $0011 the boss HP?  Count the on-screen HP-bar blocks via the OAM
shadow table at $0200 and watch $0011 while axes fly.

The HP bar is a row of identical sprites along the top of the screen; every
OAM entry is 4 bytes (y, tile, attr, x), so the number of entries sharing
the bar's y and consecutive x values IS the remaining HP.

    python tools/probe_boss_oam.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (ACTIONS, AdventureIslandEnv, N_ACTIONS,   # noqa: E402
                 OBS_DIM)
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "models", "ppo_1-3_clear_16M.pt")
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]


def oam_rows(nes):
    """group OAM sprites by (y, tile) and show the biggest rows."""
    oam = np.array([nes[0x200 + i] for i in range(256)], dtype=np.uint8)
    rows = Counter()
    for i in range(64):
        y, tile, attr, x = oam[i * 4:i * 4 + 4]
        if y < 239:
            rows[(int(y), int(tile))] += 1
    return rows.most_common(8)


def bar_count(nes, bar_y):
    """how many sprites sit on the bar's y row, x-sorted contiguity check."""
    oam = np.array([nes[0x200 + i] for i in range(256)], dtype=np.uint8)
    xs = sorted(int(oam[i * 4 + 3]) for i in range(64)
                if int(oam[i * 4]) == bar_y and int(oam[i * 4]) < 239)
    return xs


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_round=3, start_with_axe=True,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    o = env._obs()
    action, frame, best, stalls = 0, 0, 0.0, 0
    while frame < 60 * 60 * 4:
        if frame % env.frame_skip == 0:
            with torch.no_grad():
                lg, _ = net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)})
            action = int(lg[0].argmax())
        nes.controller = ACTIONS[action][0]
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        arr = np.asarray(nes.step(), dtype=np.uint8)
        nes.controller = 0
        if frame % env.frame_skip == env.frame_skip - 1:
            env._push_frame(arr)
        o = env._obs()
        frame += 1
        p = env._cam.progress(nes)
        if p > best + 0.5:
            best, stalls = p, 0
        else:
            stalls += 1
        if stalls > 45 * 60:
            break
    print(f"arena: progress={best:.0f} camera={int(nes[R.CAMERA])}", flush=True)
    print(f"$0011 = {int(nes[0x11])}")
    rows = oam_rows(nes)
    print("top OAM rows (y, tile) -> count:", rows)
    # assume the bar row is the one with the most sprites at y < 32
    bar_rows = [r for r in rows if r[0][0] < 32]
    if not bar_rows:
        print("no bar row found at y<32!")
        env.close()
        return
    (bar_y, bar_tile), _ = bar_rows[0]
    print(f"bar row: y={bar_y} tile={bar_tile}")
    print("bar x positions:", bar_count(nes, bar_y))

    # --- throw axes and watch $0011 + the bar --------------------------
    for t in range(220):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        act = A_RT if t % 6 in (0, 1) else (A_RJ if t % 19 in (0, 1) else A_RIGHT)
        env.step(act)
        if t % 10 == 0:
            xs = bar_count(nes, bar_y)
            print(f"t={t:3d}  $0011={int(nes[0x11]):3d}  "
                  f"score={int(nes[R.SCORE]):6d}  bar_blocks={len(xs)}  "
                  f"x={xs[:20]}", flush=True)
            if len(xs) == 0:
                print("bar empty!")
                break
    # final frame dump
    arr = np.asarray(nes.step(), dtype=np.uint8).copy()
    from PIL import Image
    Image.fromarray(arr).convert("RGB").resize((1024, 960), Image.NEAREST)\
        .save(os.path.join(_ROOT, "outputs", "boss_end.png"))
    print("saved outputs/boss_end.png")
    env.close()


if __name__ == "__main__":
    main()
