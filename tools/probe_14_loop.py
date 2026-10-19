"""Is 1-4 ~25000 px long, or does it LOOP?

Under real invincibility (object 199 forced into a spare slot) the 14c
policy scrolls 1-4 out to 25088 px and still has not triggered
$007A -- while 1-3 clears at roughly 6600 px.  Either 1-4 really is four
times longer than 1-3, or the level data wraps and the runner is going
round in circles.  It matters enormously:

  * if it loops  -> walking right can never clear it, the goal is wrong
  * if it is long -> MAX_STEPS (4500 env steps = 18000 game frames) is
                     too short for any policy to ever reach the end

Test: hash a coarse thumbnail of the screen every few steps and look for
the SAME thumbnail reappearing at a very different progress.  A repeating
thumbnail with a constant progress delta = a looping level.

    python tools/probe_14_loop.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, N_ACTIONS, OBS_DIM,   # noqa: E402
                 MAX_STEPS)
from train import ActorCritic                              # noqa: E402


def force_fairy(nes) -> None:
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 199:
            return
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 0:
            nes[R.OBJ.id + off] = 199
            for f in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr,
                      R.OBJ.y_lo, R.OBJ.y_fr):
                nes[f + off] = nes[f]
            return


def sig(nes, frame: np.ndarray) -> int:
    """Coarse thumbnail hash -- robust to sprites, sensitive to tiles."""
    g = frame.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    small = g[::24, ::24].astype(np.int16)
    # quantise so tiny sprite noise does not change the hash
    q = (small // 40).astype(np.int64)
    return int(hash(q.tobytes()) & 0xFFFFFFFF)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ppo_14c/ppo_final.pt")
    ap.add_argument("--round", type=int, default=3)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--steps", type=int, default=15000)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.round,
                             stage_index=args.stage)
    o = env.reset(seed=0)
    nes = env._nes

    seen: dict[int, list[float]] = {}
    for n in range(args.steps):
        force_fairy(nes)
        nes[R.STAMINA] = 10
        nes[R.LIVES] = 9
        with torch.no_grad():
            a = int(net({"ram": torch.from_numpy(o["ram"][None]),
                         "img": torch.from_numpy(o["img"][None])}
                        )[0].argmax(-1)[0])
        o, r, done, info = env.step(a)
        if n % 10 == 0:
            img = o["img"]
            fr = img[-1] if img.shape[0] <= 4 else img[..., -1]
            fr = fr * 255 if fr.max() <= 1.5 else fr
            h = sig(nes, fr)
            seen.setdefault(h, []).append(
                float(info.get("best_progress", 0.0)))
        if done:
            o = env.reset(seed=n)

    print(f"sampled {args.steps} steps; "
          f"{len(seen)} distinct screen signatures")
    # signatures seen at wildly different progress => level repeats
    repeats = []
    for h, ps in seen.items():
        if len(ps) < 2:
            continue
        spread = max(ps) - min(ps)
        if spread > 600:
            repeats.append((spread, min(ps), max(ps), len(ps), h))
    repeats.sort(reverse=True)
    print(f"\nsignatures reappearing >600 px apart: {len(repeats)}")
    for spread, lo, hi, k, h in repeats[:15]:
        print(f"  spread {spread:7.0f} px  ({lo:7.0f} -> {hi:7.0f})  "
              f"n={k}  sig={h:08X}")
    final = max(max(v) for v in seen.values())
    print(f"\nfinal best_progress seen: {final:.0f}")


if __name__ == "__main__":
    main()
