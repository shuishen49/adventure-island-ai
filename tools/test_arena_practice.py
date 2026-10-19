"""Can the current policy actually fight anything in the arena snapshot?

Starts episodes from saves/stages/stage_1-4arena.state (the 3592 px boss arena,
made by tools/make_arena_state.py) and reports, per episode, how many boss
hits ($0011 drops) it lands, how long the episode lasted and why it ended
(death / stall / clear).  Also dumps the first frame as a PNG so we can
finally *see* the boss.

    python tools/test_arena_practice.py --checkpoint runs/ppo_14c/ppo_*.pt \
        --episodes 4 --greedy
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

from env import (AdventureIslandEnv, OBS_DIM, N_ACTIONS,   # noqa: E402
                 MAX_STEPS)
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "runs", "ppo_14", "ppo_final.pt")
ARENA = os.path.join(_ROOT, "saves", "stages", "stage_1-4arena.state")
OUT = os.path.join(_ROOT, "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--state", default=ARENA)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--stage-idx", type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=args.stage_idx,
                             start_state=args.state,
                             clear_ends_episode=True)

    for e in range(args.episodes):
        o = env.reset(seed=6000 + e)
        done, n = False, 0
        hits, deaths = 0, 0
        hp0 = int(env._nes[0x0011])
        info = {}
        first = None
        while not done and n < MAX_STEPS:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax()) if args.greedy else int(
                torch.distributions.Categorical(logits=lg[0]).sample())
            o, r, done, info = env.step(a)
            n += 1
            hits += info.get("boss_hits", 0)
            if info.get("death"):
                deaths += 1
            if first is None:
                first = env._last_frame.copy()

        why = ("通关" if info.get("cleared") else
               "死亡" if info.get("death") else
               "卡住" if info.get("stalled") else "步数上限")
        print(f"  ep{e + 1:2d}: {n:4d} 步  boss 命中 {hits}  "
              f"$0011 {hp0} -> {info.get('boss_hp')}  "
              f"死亡 {deaths}  结束原因 {why}  "
              f"best_progress={info.get('best_progress', 0):.0f}")
        if first is not None and e == 0:
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(OUT, "arena_start.png"), first)
            print("     首帧已存 outputs/arena_start.png")


if __name__ == "__main__":
    main()
