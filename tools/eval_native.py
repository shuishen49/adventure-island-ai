"""Evaluate a checkpoint INSIDE the real training env (ground truth).

evaluate.py rebuilds the observation from a duck-typed shim, which is
handy for a quick look but is free to drift away from what the workers
actually feed the net.  When the two disagree, this script is the referee:
it drives ``AdventureIslandEnv`` itself, so every RAM feature, the frame
stack and the stage one-hot are produced by exactly the same code path
that produced the numbers in runs/*/progress.csv.

    python tools/eval_native.py runs/ppo_12/ppo_final.pt \
        --start-state saves/stages/stage_1-2.state --stage-idx 1 --episodes 12
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from env import (AdventureIslandEnv, OBS_DIM, N_ACTIONS, MAX_STEPS)  # noqa: E402
from train import ActorCritic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--stage-idx", type=int, default=0,
                    help="one-hot tail.  If the checkpoint was trained with "
                         "--stages, this must be the POSITION OF --start-state "
                         "IN THAT POOL (sorted 1-1,1-2,...), e.g. 3 for 1-4 "
                         "and 4 for 1-4boss in pool 1-1,1-2,1-3,1-4,1-4boss -- NOT "
                         "area*4+round.")
    ap.add_argument("--no-axe", action="store_true",
                    help="do NOT pre-equip the axe (workers do by default)")
    ap.add_argument("--clear-ends-episode", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(
        start_with_axe=not args.no_axe,
        start_round=args.start_round,
        start_state=args.start_state,
        stage_index=args.stage_idx,
        clear_ends_episode=args.clear_ends_episode)

    xs, steps, kills, bests = [], [], [], []
    for e in range(args.episodes):
        o = env.reset(seed=1000 + e)
        n = 0
        done = False
        while not done and n < args.max_steps:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax()) if args.greedy else int(
                torch.distributions.Categorical(logits=lg[0]).sample())
            o, r, done, info = env.step(a)
            n += 1
        xs.append(float(info["x_progress"]))
        steps.append(n)
        kills.append(int(info.get("kills", 0)))
        cleared = bool(info.get("cleared", False))
        # x_progress is the SUM of the small forward deltas only: any jump
        # of |dp| >= 64 (respawn, but also a rapid on-screen bounce) is
        # dropped, so an agent that oscillates forward/back banks the
        # return leg and is never charged for the outbound one.  On 1-4
        # that inflates the number ~2.6x (measured 2026-10-17: x_progress
        # 1911 px on a run whose true maximum was 744 px).  best_progress
        # is the honest "furthest point actually reached".
        bests.append(float(info.get("best_progress", 0.0)))
        print(f"  ep{e+1:2d}: 真实 {info['best_progress']:7.1f} px  "
              f"(累加 {info['x_progress']:7.1f})  "
              f"{n:4d} 决策 ({n * env.frame_skip} 帧)  "
              f"死 {int(info['death'])}  吃道具 {info.get('score', 0)}  "
              f"杀敌 {int(info.get('kills', 0))}"
              + ("   *** CLEAR ***" if cleared else ""))

    xs = np.asarray(xs)
    bs = np.asarray(bests)
    print()
    print(f"真实进度  平均 {bs.mean():.1f} px  中位 {np.median(bs):.1f}  "
          f"最好 {bs.max():.1f}  最差 {bs.min():.1f}")
    print(f"累加口径  平均 {xs.mean():.1f} px  中位 {np.median(xs):.1f}  "
          f"最好 {xs.max():.1f}  最差 {xs.min():.1f}   "
          f"平均 {np.mean(steps):.0f} 决策/局")


if __name__ == "__main__":
    main()
