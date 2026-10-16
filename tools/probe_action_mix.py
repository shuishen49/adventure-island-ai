"""Action-mix profile of a checkpoint: does the policy ever throw (B),
does it pick up the axe, and where does it die?

Runs N greedy (argmax) episodes with the real env and reports:
  - per-episode progress and deaths
  - histogram over the 8 discrete actions
  - axe pickups / frames spent armed / throw presses

Usage:  python tools/probe_action_mix.py <ckpt> [--episodes N]
"""

from __future__ import annotations

import argparse
import os
import sys

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
    ap.add_argument("--max-steps", type=int, default=4500)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv()
    frame_skip = env.frame_skip
    nes = env._nes
    counts = np.zeros(len(ACTIONS), dtype=np.int64)
    axe_pickups = 0
    skate_pickups = 0
    throws = 0
    armed_frames = 0
    ep_prog, ep_deaths, ep_steps = [], [], []

    for ep in range(args.episodes):
        obs = env.reset(seed=100 + ep)
        steps = 0
        deaths = 0
        best_x = 0.0
        prev_lives = nes[R.LIVES]
        while steps < args.max_steps:
            with torch.no_grad():
                ram = torch.as_tensor(obs["ram"], dtype=torch.float32).unsqueeze(0)
                img = torch.as_tensor(obs["img"], dtype=torch.float32).unsqueeze(0)
                logits = net({"ram": ram, "img": img})[0]
            a = int(logits.argmax())
            counts[a] += 1
            if "throw" in ACTIONS[a][1]:
                throws += 1
            obs, rew, term, info = env.step(a)
            steps += 1
            if nes[R.WEAPON] > 0:
                armed_frames += 1
            if info.get("got_axe"):
                axe_pickups += 1
            if info.get("got_skate"):
                skate_pickups += 1
            if nes[R.LIVES] < prev_lives:
                deaths += 1
                prev_lives = nes[R.LIVES]
                if nes[R.LIVES] == 0:
                    break
            best_x = max(best_x, float(info.get("x_progress", 0)))
        ep_prog.append(best_x)
        ep_deaths.append(deaths)
        ep_steps.append(steps)

    env.close()

    total = counts.sum()
    print(f"--- {args.episodes} episodes, {total} decisions")
    print("    action mix:")
    for i, (mask, name) in enumerate(ACTIONS):
        bar = "#" * int(40 * counts[i] / max(1, total))
        print(f"      {i} {name:12s} {counts[i]:7d} "
              f"({100 * counts[i] / max(1, total):5.1f}%) {bar}")
    print(f"    throw presses : {throws} "
          f"({100 * throws / max(1, total):.1f}% of decisions)")
    print(f"    axe pickups   : {axe_pickups}   skateboard pickups: {skate_pickups}")
    print(f"    armed steps   : {armed_frames} "
          f"({100 * armed_frames / max(1, total):.1f}% of decisions)")
    print("    per episode (progress / deaths / steps):")
    for i, (p, d, s) in enumerate(zip(ep_prog, ep_deaths, ep_steps)):
        print(f"      ep{i}: {p:8.1f} px   deaths {d}   steps {s}")


if __name__ == "__main__":
    main()
