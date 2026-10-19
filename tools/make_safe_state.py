"""Capture a SAFE restart state near a given progress in 1-4.

The previous states (stage_1-4_deep.state at 5506 px and
stage_1-4_5400.state) were dumped on the exact frame the runner *reached*
that progress -- which turned out to be mid-fall over a pit, so every
reload died within 2 steps.  This one keeps playing past the target and
only saves on a frame where the runner is

  * standing on ground (y barely moved over the last frames),
  * not touching an enemy (nothing within 55 px horizontally / 40 px
    vertically, except our own axe id 66),
  * has a decent stamina buffer.

No cheats: the policy plays for real and the state is a genuine frame the
policy reached on its own.

    python tools/make_safe_state.py --target 5400
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

# our own axe is harmless; everything else in ENEMY_IDS is not
SAFE_IDS = set(R.ENEMY_IDS) | {5, 67, 198}
AXE_ID = 66


def safe_now(nes, prev_y: float | None, min_stamina: int,
             require_boss: bool = False,
             dx_lo: float = 60.0, dx_hi: float = 170.0
             ) -> tuple[bool, str]:
    if int(nes[R.DEATH]) != 0:
        return False, "dying"
    if int(nes[R.STAMINA]) < min_stamina:
        return False, f"stamina {int(nes[R.STAMINA])}"
    py = R.px(R.read_obj_world_y(nes, 0))
    px = R.px(R.read_obj_world_x(nes, 0))
    if prev_y is not None and abs(py - prev_y) > 1.5:
        return False, f"falling ({prev_y:.0f}->{py:.0f})"
    for s in range(1, 16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0 or oid == AXE_ID:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        dx, dy = ox - px, oy - py
        if abs(dx) < 55 and abs(dy) < 40:
            return False, f"id{oid} at dx{dx:+.0f} dy{dy:+.0f}"
    if require_boss:
        # the boss must already be loaded (so a reload starts the fight at
        # once) but far enough away that we are not already inside it
        head_dx = None
        for s in range(1, 16):
            oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
            if oid in R.BOSS_IDS:
                ox = R.px(R.read_obj_world_x(nes, s))
                d = ox - px
                if head_dx is None or abs(d) < abs(head_dx):
                    head_dx = d
        if head_dx is None:
            return False, "no boss"
        if not (dx_lo <= head_dx <= dx_hi):
            return False, f"boss dx{head_dx:+.0f} out of range"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=os.path.join(_ROOT, "runs", "ppo_14d",
                                         "ppo_27770880.pt"))
    ap.add_argument("--target", type=float, default=5400.0)
    ap.add_argument("--min-progress", type=float, default=None,
                    help="accept any frame at or beyond this "
                         "(default: target - 120)")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--min-stamina", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_safe.state"))
    ap.add_argument("--round", type=int, default=3,
                    help="0-based $0038 (3 = 1-4, 4 = 2-1)")
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--require-boss", action="store_true",
                    help="only save frames where the boss (67/198) "
                         "is already loaded and 60..170 px ahead")
    ap.add_argument("--dx-lo", type=float, default=60.0)
    ap.add_argument("--dx-hi", type=float, default=170.0)
    args = ap.parse_args()

    lo = args.min_progress if args.min_progress is not None \
        else args.target - 120.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"model={args.checkpoint}  target>={lo:.0f}px  -> {args.out}",
          flush=True)

    best = None
    for ep in range(args.episodes):
        env = AdventureIslandEnv(start_with_axe=True, start_round=args.round,
                                 stage_index=args.stage_idx)
        env.reset(seed=1000 + ep)
        nes = env._nes
        prev_y = None
        reached = False
        for step in range(4500):
            o = env._obs()
            with torch.no_grad():
                a = int(net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)}
                            )[0].argmax())
            o, r, done, info = env.step(a)
            p = info.get("best_progress", 0.0)
            py = R.px(R.read_obj_world_y(nes, 0))
            if p >= lo:
                reached = True
                ok, why = safe_now(nes, prev_y, args.min_stamina,
                                   args.require_boss,
                                   args.dx_lo, args.dx_hi)
                if ok:
                    # prefer the frame closest to (but not past) the target
                    score = -abs(p - args.target)
                    if best is None or score > best[0]:
                        best = (score, p, bytes(nes.save()), ep, step)
                        print(f"ep{ep}: SAFE frame at {p:.0f} px "
                              f"(step {step}) stamina="
                              f"{int(nes[R.STAMINA])}", flush=True)
                        if abs(p - args.target) <= 25:
                            break
            prev_y = py
            if done:
                break
        if best is not None and abs(best[1] - args.target) <= 25:
            break
        if not reached and ep % 5 == 0:
            print(f"  ...ep{ep} never reached {lo:.0f} px", flush=True)

    if best is None:
        print("no safe frame found; try a lower --min-progress")
        return
    score, p, blob, ep, step = best
    with open(args.out, "wb") as fh:
        fh.write(blob)
    print(f"\nsaved {p:.0f} px (ep{ep} step{step}) -> {args.out}")


if __name__ == "__main__":
    main()
