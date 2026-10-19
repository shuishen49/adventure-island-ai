#!/usr/bin/env python
"""Capture a frame BEFORE the boss arena -- but not right on top of it.

Why this exists
---------------
saves/stages/stage_1-4gate.state is the first frame where the boss is on
screen and Higgins is grounded: the boss sits ~95 px ahead, i.e. already
inside axe range (range ~57 px, safe band ~25..70 px).  Watching that is
useless for judging whether the policy can *approach* the arena: it never
has to walk, it just has to stop.

This tool runs the same check-point from an earlier snapshot and keeps a
ring buffer of the last N frames.  The moment the boss spawns it walks the
buffer BACKWARDS and dumps the newest frame where the (estimated) distance
to the boss is still >= --target-dx and Higgins stands on the floor, so the
window shows "run in -> keep the range -> jump-throw" from further out.

The distance before the spawn is estimated as
    dx_est = dx_at_spawn + (player_x_at_spawn - player_x_now)
because the boss object is not in the table yet; player x only moves
forward, so the estimate is exact as long as the run did not wrap.

usage:
  python tools/make_approach_state.py \
      --checkpoint runs/ppo_14l/ppo_42516480.pt \
      --from-state saves/stages/stage_1-4late.state \
      --out saves/stages/stage_1-4approach.state --target-dx 260
"""
import argparse
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R                                          # noqa: E402
from env import AdventureIslandEnv, N_ACTIONS, OBS_DIM       # noqa: E402

GROUND_Y = 140          # player y_lo when standing on the arena floor


def _signed(a: int, b: int) -> int:
    """a - b for 8-bit wrap-around registers."""
    return ((a - b + 128) % 256) - 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--from-state",
                    default=os.path.join("saves", "stages",
                                         "stage_1-4late.state"))
    ap.add_argument("--out", default=os.path.join("saves", "stages",
                                                  "stage_1-4approach.state"))
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--target-dx", type=int, default=260,
                    help="want the boss about this far ahead of Higgins")
    ap.add_argument("--buffer", type=int, default=900,
                    help="how many pre-spawn frames to keep")
    ap.add_argument("--ground-y-min", type=int, default=138)
    ap.add_argument("--ground-y-max", type=int, default=148,
                    help="only accept a frame with Higgins standing on the "
                         "floor (y_lo in this band); a frame deeper than "
                         "that is already falling into a pit and dies on "
                         "restore")
    args = ap.parse_args()

    from train import ActorCritic
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    ID, XL, YL = R.OBJ.id, R.OBJ.x_lo, R.OBJ.y_lo
    best = None
    for ep in range(args.episodes):
        env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                                 stage_index=args.stage_idx,
                                 start_state=args.from_state)
        o = env.reset(seed=9000 + ep)
        nes = env._nes
        buf: deque = deque(maxlen=args.buffer)
        for n in range(args.max_steps):
            with torch.no_grad():
                a = int(net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)}
                            )[0].argmax())
            buf.append((env._cam.progress(nes), int(nes[YL]),
                       int(nes[R.STAMINA]), bytes(nes.save())))
            o, _, done, info = env.step(a)
            slots = [s for s in range(1, 16)
                     if int(nes[ID + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID]
            if not slots:
                if done:
                    break
                continue
            appear_pr = env._cam.progress(nes)
            dx = int(nes[XL + R.OBJ.STRIDE * slots[0]]) - int(nes[XL])
            print(f"  ep{ep} step{n}: boss spawned, dx={dx}, "
                  f"progress={appear_pr}, y={int(nes[YL])}, "
                  f"stamina={int(nes[R.STAMINA])}, buffer={len(buf)}",
                  flush=True)
            # walk back to the newest frame still far enough away
            for pr, py, st, blob in reversed(buf):
                dx_est = dx + (appear_pr - pr)
                if dx_est >= args.target_dx and (args.ground_y_min
                                                 <= py <= args.ground_y_max):
                    best = (dx_est, py, st, int(pr), blob)
                    break
            break
        env.close()
        if best:
            break

    if best is None:
        print("no pre-spawn frame far enough away; try a smaller "
              "--target-dx or a --buffer large enough")
        return
    dx, py, st, pr, blob = best
    with open(args.out, "wb") as fh:
        fh.write(blob)
    print(f"saved {args.out}: boss ~{dx} px ahead (progress {pr}), "
          f"player y={py}, stamina={st} ({len(blob)} bytes)")


if __name__ == "__main__":
    main()
