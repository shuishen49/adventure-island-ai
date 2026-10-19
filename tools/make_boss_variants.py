#!/usr/bin/env python
"""Collect MANY different boss-fight frames, not just the one practice shot.

The diagnosis (2026-10-19): the policy kills the boss 8/10 from
stage_1-4boss.state -- the single hand-made arena snapshot it trained on --
and scores exactly 0 hits from every REAL arrival frame (1-4gate, 1-4gate79,
1-4approach, 1-4late), for every checkpoint we have.  A scripted bot does
hit the head 5 times from 1-4gate79, so those frames are winnable; the
policy is simply over-fitted to one picture (one camera value, one head
position, one object layout).

Fix: train on a spread of real fight frames so "boss on screen" stops being
a single memorised image.  This tool replays a checkpoint into the real 1-4
arena and dumps every Nth frame in which
    * the boss head is loaded,
    * Higgins stands on the floor (not falling into a pit),
    * nobody is dying right now,
keeping only frames whose (camera, dx) pair is new.

usage:
  python tools/make_boss_variants.py --checkpoint runs/ppo_14l/X.pt \
      --out-prefix 1-4bv --count 8
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

import ram_map as R                                          # noqa: E402
from env import AdventureIslandEnv, N_ACTIONS, OBS_DIM       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--from-state", default=os.path.join(
        "saves", "stages", "stage_1-4late.state"))
    ap.add_argument("--out-prefix", default="1-4bv")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--every", type=int, default=25,
                    help="sample at most one frame every N decisions")
    ap.add_argument("--dx-min", type=int, default=35)
    ap.add_argument("--dx-max", type=int, default=130)
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=2500)
    args = ap.parse_args()

    from train import ActorCritic
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    ID, XL, YL = R.OBJ.id, R.OBJ.x_lo, R.OBJ.y_lo
    root = os.path.join("saves", "stages")
    seen: set[tuple[int, int]] = set()
    saved: list[tuple[str, int, int, int]] = []

    for ep in range(args.episodes):
        if len(saved) >= args.count:
            break
        env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                                 stage_index=args.stage_idx,
                                 start_state=args.from_state)
        o = env.reset(seed=3000 + ep)
        nes = env._nes
        last = -10 ** 9
        for n in range(args.max_steps):
            with torch.no_grad():
                a = int(net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)}
                            )[0].argmax())
            o, _, done, info = env.step(a)
            slots = [s for s in range(1, 16)
                     if int(nes[ID + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID]
            if not slots or int(nes[R.DEATH]):
                if done:
                    break
                continue
            if n - last < args.every:
                continue
            px, py = int(nes[XL]), int(nes[YL])
            dx = int(nes[XL + R.OBJ.STRIDE * slots[0]]) - px
            cam = int(nes[R.CAMERA])
            if not (args.dx_min <= dx <= args.dx_max and 138 <= py <= 148):
                continue
            key = (cam, dx // 8)
            if key in seen:
                continue
            seen.add(key)
            last = n
            name = f"{args.out_prefix}{len(saved) + 1}"
            path = os.path.join(root, f"stage_{name}.state")
            with open(path, "wb") as fh:
                fh.write(bytes(nes.save()))
            saved.append((name, dx, cam, int(nes[R.STAMINA])))
            print(f"  ep{ep} n={n}: {name}  dx={dx} cam={cam} "
                  f"y={py} stamina={int(nes[R.STAMINA])}", flush=True)
            if len(saved) >= args.count:
                break
        env.close()

    print(f"\n{len(saved)} frames:")
    for name, dx, cam, st in saved:
        print(f"  {name}: dx={dx} camera={cam} stamina={st}")
    if saved:
        pool = ",".join(f"{n}:0.06@3" for n, *_ in saved)
        print(f"\npool fragment -> {pool}")


if __name__ == "__main__":
    main()
