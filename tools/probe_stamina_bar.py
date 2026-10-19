"""Where is the player's stamina bar, and does it survive the boss fight?

The user says the watch window "has no health bar".  I cannot look at
pictures, so this measures it:

  1. locate the bar by CAUSATION -- write STAMINA = 10, render, then
     STAMINA = 2, render, and diff.  The pixels that change ARE the bar.
  2. repeat the same diff once the boss (id 67 / 198) is on screen.  If the
     bar is gone there, the diff will be empty.

    python tools/probe_stamina_bar.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, ACTIONS, N_ACTIONS,    # noqa: E402
                 OBS_DIM)
from train import ActorCritic                              # noqa: E402


def frame_of(nes, stamina: int, settle: int = 3) -> np.ndarray:
    nes[R.STAMINA] = stamina
    for _ in range(settle):
        fr = np.asarray(nes.step(1), dtype=np.uint8)
    return fr


def diff_box(a: np.ndarray, b: np.ndarray):
    g1 = a.astype(np.int16).mean(axis=2)
    g2 = b.astype(np.int16).mean(axis=2)
    d = np.abs(g1 - g2) > 30
    if not d.any():
        return None
    ys, xs = np.nonzero(d)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(d.sum())


def boss_on(nes) -> bool:
    return any(int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) in R.BOSS_IDS
               for s in range(1, 16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_safe5400.state"))
    ap.add_argument("--checkpoint", default=os.path.join(
        _ROOT, "runs", "ppo_14d", "ppo_27770880.pt"))
    args = ap.parse_args()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3, start_state=args.state)
    env.reset(seed=0)
    nes = env._nes

    print("--- normal play (boss not on screen yet) ---", flush=True)
    snap = nes.save()
    hi = frame_of(nes, 10)
    nes.load(snap)
    lo = frame_of(nes, 2)
    box = diff_box(hi, lo)
    print(f"  STAMINA 10 vs 2 -> {box}", flush=True)

    nes.load(snap)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    boss_snap = None
    for _ in range(300):
        o = env._obs()
        with torch.no_grad():
            a = int(net({"img": torch.as_tensor(o["img"][None]),
                         "ram": torch.as_tensor(o["ram"][None],
                                                dtype=torch.float32)}
                        )[0].argmax())
        o, r, done, info = env.step(a)
        if boss_on(nes):
            boss_snap = nes.save()
            break
        if done:
            break
    if boss_snap is None:
        print("  never reached the boss")
        return

    print("--- boss on screen ---", flush=True)
    nes.load(boss_snap)
    hi2 = frame_of(nes, 10)
    nes.load(boss_snap)
    lo2 = frame_of(nes, 2)
    box2 = diff_box(hi2, lo2)
    print(f"  STAMINA 10 vs 2 -> {box2}", flush=True)
    if box2 is None:
        print("  => the stamina bar is NOT drawn any more in the boss fight")
    else:
        print("  => the stamina bar is still drawn")


if __name__ == "__main__":
    main()
