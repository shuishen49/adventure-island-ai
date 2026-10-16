"""Smoke test for the axe reward: does "picked up the weapon" ever fire?

Runs a few episodes with a run-right-and-hop policy and reports every
episode's reward breakdown.  Also checks the observation vector.

Usage:  python tools/check_axe_reward.py [episodes]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import AdventureIslandEnv  # noqa: E402

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def main() -> None:
    env = AdventureIslandEnv(seed=0)
    obs = env.reset()
    print(f"ram obs dim = {obs['ram'].shape}  img = {obs['img'].shape}")
    print("last 3 ram values = nearest item (dx, dy, seen):",
          np.round(obs["ram"][-3:], 3))

    rng = np.random.default_rng(1)
    for ep in range(N_EP):
        obs = env.reset()
        tot = 0.0
        items_seen = 0
        axe_bonus = 0.0
        while True:
            # chase the nearest power-up: this is what we want the agent to
            # learn, so the bonus has to be reachable with a simple policy
            i_dx, i_dy, i_seen = obs["ram"][-3:]
            if i_seen and abs(i_dx) < 2.0:
                a = 1 if i_dx > 0 else 2
                if i_dy < -0.05:               # item is above: hop
                    a = 4 if i_dx > 0 else 5
            else:
                a = 4 if rng.random() < 0.5 else 1
            if rng.random() < 0.15:
                a = 7 if rng.random() < 0.5 else 6
            obs, r, done, info = env.step(a)
            tot += r
            if info["got_axe"]:
                axe_bonus += 1
            if obs["ram"][-1] > 0:
                items_seen += 1
            if done:
                break
        print(f"ep{ep}: steps={env._steps:4d} reward={tot:7.2f} "
              f"x={info['x_progress']:7.0f} score={info['score']:3d} "
              f"weapon={info['weapon']} pickups={axe_bonus} "
              f"frames_with_item={items_seen} death={info['death']}")
    env.close()


if __name__ == "__main__":
    main()
