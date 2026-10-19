"""Score a checkpoint on the BOSS fight, not on how far it walks.

The usual eval reports best_progress, which is exactly the wrong number for
the 1-4 boss: once the boss is on screen the progress reward is switched off
(the user asked for that), so an agent that fights well can look worse than
one that runs past and dies.  This script reports what matters:

    hits   - head hits landed (0..8; 8 = the boss dies)
    kills  - how many episodes actually finished the boss
    survive- how long the agent lasted in the arena

    python tools/eval_boss.py runs/ppo_14d/ppo_27770880.pt --episodes 12

Stage "1-4boss" is saves/stages/stage_1-4boss.state: the arena snapshot
with the boss already 91 px ahead.  "1-4late" is stage_1-4late.state, a
mid-1-4 restart (~3500 px).  The game has no 1-5/1-6/1-7 -- those were
bad names for practice snapshots and are gone (2026-10-18).
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

from env import (AdventureIslandEnv, N_ACTIONS, OBS_DIM,   # noqa: E402
                 load_state_file)
from train import ActorCritic                              # noqa: E402


def run(net, state: str, episodes: int, seed0: int, max_steps: int,
        stage_idx: int = 4):
    hits, kills, steps_live, best = [], 0, [], []
    for ep in range(episodes):
        # stage_index MUST be the position of `state` inside the training
        # stage pool (env.reset sets _stage_idx = i), otherwise the 32-way
        # one-hot tells the policy it is on a different stage and it plays
        # badly for no reason.  Pool "1-1,1-2,1-3,1-4,1-4boss,1-4late"
        # -> 1-4 is 3, 1-4boss is 4, 1-4late is 5.
        env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                                 stage_index=stage_idx, start_state=state)
        o = env.reset(seed=seed0 + ep)
        h = 0
        n = 0
        for _ in range(max_steps):
            with torch.no_grad():
                a = int(net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)}
                            )[0].argmax())
            o, r, done, info = env.step(a)
            h += int(info.get("boss_hits", 0))
            n += 1
            if info.get("boss_killed"):
                kills += 1
                break
            if done:
                break
        hits.append(h)
        steps_live.append(n)
        best.append(float(info.get("best_progress", 0.0)))
        env.close()
    return np.array(hits), kills, np.array(steps_live), np.array(best)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--boss-state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4boss.state"))
    ap.add_argument("--travel-state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4.state"))
    ap.add_argument("--skip-travel", action="store_true")
    ap.add_argument("--stages", default="1-1:0.02,1-2:0.03,1-3:0.05,"
                    "1-4:0.33,1-4boss:0.25,1-4late:0.32",
                    help="the training curriculum, used ONLY to work out "
                         "the one-hot index of --boss-state / "
                         "--travel-state (pool position, canonical order)")
    ap.add_argument("--boss-idx", type=int, default=None)
    ap.add_argument("--travel-idx", type=int, default=None)
    args = ap.parse_args()

    def pool_index(path: str, fallback: int) -> int:
        from env import parse_stage_pool
        _, _, names, idxs = parse_stage_pool(args.stages)
        tag = os.path.basename(path)[6:-6]    # stage_1-4boss.state -> 1-4boss
        # honour an explicit "@N" override, else the pool position
        return idxs[names.index(tag)] if tag in names else fallback

    boss_idx = (args.boss_idx if args.boss_idx is not None
                else pool_index(args.boss_state, 4))      # 1-4boss
    travel_idx = (args.travel_idx if args.travel_idx is not None
                  else pool_index(args.travel_state, 3))  # 1-4
    print(f"one-hot index: boss={boss_idx} travel={travel_idx}")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    h, k, live, prog = run(net, args.boss_state, args.episodes, 500,
                           args.max_steps, boss_idx)
    print(f"[boss arena] {args.checkpoint}")
    print(f"  hits per episode : {h.tolist()}")
    print(f"  mean hits        : {h.mean():.2f}  (max {h.max()})")
    print(f"  boss killed      : {k}/{args.episodes}")
    print(f"  mean survival    : {live.mean():.0f} steps")
    print(f"  best_progress    : {prog.max():.0f} px")

    if not args.skip_travel:
        h2, k2, live2, prog2 = run(net, args.travel_state, args.episodes,
                                   900, args.max_steps, travel_idx)
        print(f"[1-4 from the start]")
        print(f"  best_progress    : mean {prog2.mean():.0f} px, "
              f"best {prog2.max():.0f} px")
        print(f"  boss hits        : {h2.tolist()}  kills {k2}")


if __name__ == "__main__":
    main()
