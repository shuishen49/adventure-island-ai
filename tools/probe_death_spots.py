"""Where does the policy actually die?  Runs greedy episodes and logs the
progress value at every death, plus the player's Y so we can tell a pit
fall (y runs off the bottom) from an enemy hit.

Usage:  python tools/probe_death_spots.py <ckpt> [--episodes N]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R                     # noqa: E402
from env import ACTIONS, AdventureIslandEnv, OBS_DIM  # noqa: E402
from train import ActorCritic           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv()
    nes = env._nes
    deaths: list[tuple[float, int, str]] = []   # (progress, y, cause)
    furthest = []

    for ep in range(args.episodes):
        obs = env.reset(seed=200 + ep)
        best = 0.0
        prev_lives = nes[R.LIVES]
        while env._steps < 4500:
            with torch.no_grad():
                ram = torch.as_tensor(obs["ram"], dtype=torch.float32).unsqueeze(0)
                img = torch.as_tensor(obs["img"], dtype=torch.float32).unsqueeze(0)
                logits = net({"ram": ram, "img": img})[0]
            a = int(logits.argmax())
            # state right before the step that may kill us
            px = float(env._x_progress)
            py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
            sta = nes[R.STAMINA]
            obs, rew, term, info = env.step(a)
            best = max(best, float(info["x_progress"]))
            lv = nes[R.LIVES]
            if lv < prev_lives:                      # exactly one row per death
                cause = ("PIT" if py > 190
                         else "STARVE" if sta <= 1
                         else "HIT")
                deaths.append((round(px), int(py), cause))
                prev_lives = lv
                if lv == 0:
                    break
        furthest.append(best)

    env.close()

    print(f"--- {len(deaths)} deaths over {args.episodes} episodes")
    print("    death spots (progress px, player y, cause):")
    for x, y, c in deaths:
        print(f"      x={x:6d}  y={y:3d}  {c}")
    buckets = Counter(x // 64 * 64 for x, _, _ in deaths)
    print("    hotspots (64 px buckets):")
    for b, n in buckets.most_common(8):
        print(f"      x {b:5d}-{b + 63:5d}: {n} deaths")
    print("    per-episode furthest:", [f"{v:.0f}" for v in furthest])


if __name__ == "__main__":
    main()
