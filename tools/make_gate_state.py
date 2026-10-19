#!/usr/bin/env python
"""Capture the moment the boss appears during a REAL 1-4 run.

Why this exists
---------------
saves/stages/stage_1-4boss.state is a hand-made arena snapshot: Higgins
stands still, stamina 10, the boss 91 px ahead.  That is NOT what the
policy actually looks like when it arrives at the arena at the end of a
full 1-4 run -- there it comes in running, usually mid-jump, with the
boss ~130 px away and stamina ~6 (measured 2026-10-18: arrival frame
dx=131..136, y=82..144, stamina 6/10).

Because the arrival used to be the last frame of the episode (a bug in
the stall clock -- see env.BOSS_STALL_STEPS), the policy had never
practised that transition, and a greedy run walked straight from dx=131
into the boss without ever throwing an axe.

This tool replays a checkpoint from stage_1-4.state and dumps the first
frame where the boss is on screen and Higgins is grounded, so training
can rehearse "arrive -> keep the range -> jump-throw" directly instead
of paying 1200 steps of travel for one attempt.

usage:
  python tools/make_gate_state.py --checkpoint runs/ppo_14h/ppo_35758080.pt \
         --out saves/stages/stage_1-4gate.state
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R                                          # noqa: E402
from env import (AdventureIslandEnv, N_ACTIONS, OBS_DIM,     # noqa: E402
                 NES_INPUT_RIGHT, NES_INPUT_A)

GROUND_Y = 140          # player y_lo when standing on the arena floor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--from-state",
                    default=os.path.join("saves", "stages",
                                         "stage_1-4.state"))
    ap.add_argument("--out", default=os.path.join("saves", "stages",
                                                  "stage_1-4gate.state"))
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=4500)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--min-dx", type=int, default=100,
                    help="want the boss at least this far ahead, so the "
                         "policy has room to approach and line up a throw")
    ap.add_argument("--scripted", action="store_true",
                    help="ignore the checkpoint: just run right and jump, "
                         "with death/starvation frozen (fallback)")
    args = ap.parse_args()

    net = None
    if not args.scripted:
        from train import ActorCritic
        ck = torch.load(args.checkpoint, map_location="cpu",
                        weights_only=False)
        net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
        net.load_state_dict(ck["model"])
        net.eval()

    ID, XL, YL = R.OBJ.id, R.OBJ.x_lo, R.OBJ.y_lo
    best = None
    for ep in range(args.episodes):
        env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                                 stage_index=args.stage_idx,
                                 start_state=args.from_state)
        o = env.reset(seed=7000 + ep)
        nes = env._nes
        for n in range(args.max_steps):
            if args.scripted:
                nes[R.STAMINA] = 10
                nes[R.DEATH] = 0
                nes.controller = (NES_INPUT_RIGHT
                                  | (NES_INPUT_A if n % 26 < 5 else 0))
                nes.step()
                nes.controller = 0
                done = False
                info = {}
            else:
                with torch.no_grad():
                    a = int(net({"img": torch.as_tensor(o["img"][None]),
                                 "ram": torch.as_tensor(
                                     o["ram"][None], dtype=torch.float32)}
                                )[0].argmax())
                o, _, done, info = env.step(a)
            slots = [s for s in range(1, 16)
                     if int(nes[ID + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID]
            if not slots:
                if done:
                    break
                continue
            px = int(nes[XL])
            dx = int(nes[XL + R.OBJ.STRIDE * slots[0]]) - px
            py = int(nes[YL])
            print(f"  ep{ep} step{n}: boss on screen, dx={dx}, "
                  f"player y={py}, stamina={int(nes[R.STAMINA])}",
                  flush=True)
            if py >= GROUND_Y and dx >= args.min_dx:
                best = (dx, py, ep, bytes(nes.save()))
                break
            if done:
                break
        env.close()
        if best:
            break

    if best is None:
        print("no grounded arrival frame with the boss far enough; "
              "try --min-dx lower or --scripted")
        return
    dx, py, ep, blob = best
    with open(args.out, "wb") as fh:
        fh.write(blob)
    print(f"saved {args.out}: boss {dx} px ahead, player y={py} "
          f"(ep{ep}, {len(blob)} bytes)")


if __name__ == "__main__":
    main()
