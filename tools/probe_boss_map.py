"""Map $0011 -> the on-screen HP bar, one poke at a time.

$0011 stepped down 8 -> 1 while axes were landing, but poking it to 0 did
NOT end the fight.  So either it is not the HP counter, or the bar is not a
plain function of it.  Writing every value 0..14 and reading the bar back
settles it: if the bar length tracks the poke, we have our byte.

    python tools/probe_boss_map.py
"""
from __future__ import annotations

import os
import sys

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


def bar_len(frame: np.ndarray) -> int:
    strip = frame[36:46].astype(np.int16).mean(axis=2)
    cols = (strip > 120)[:, 75:150].any(axis=0)
    runs, in_run = 0, False
    for c in cols:
        if c and not in_run:
            runs += 1
            in_run = True
        elif not c:
            in_run = False
    return runs


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
    print(f"arena: progress={best:.0f}  $0011={int(nes[0x11])}", flush=True)

    print("poke $0011 -> bar length")
    for k in range(0, 15):
        nes[0x11] = k
        for _ in range(4):
            nes[R.DEATH] = 0
            nes[R.STAMINA] = 10
            env.step(0)
        f = np.asarray(nes.step(), dtype=np.uint8).copy()
        bl = bar_len(f)
        kept = int(nes[0x11])
        print(f"  $0011={k:2d} -> bar={bl:2d}   (register now {kept})"
              f"  STAGE_COMPLETE={int(nes[R.STAGE_COMPLETE])}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
