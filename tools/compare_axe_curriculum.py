"""Does 'start every episode with the axe' actually speed up learning?

Runs the SAME checkpoint through the env twice -- once with the normal
game (no axe until the first egg is collected) and once with the axe
armed at reset -- and compares how far it gets and how long it survives.

    python tools/compare_axe_curriculum.py runs/ppo_v2/ppo_00399360.pt
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from env import OBS_DIM, AdventureIslandEnv
from train import ActorCritic


def run(net, start_with_axe: bool, episodes: int = 6, max_steps: int = 1500
        ) -> dict:
    env = AdventureIslandEnv(start_with_axe=start_with_axe)
    xs, lens, axes, kills = [], [], [], 0
    for _ in range(episodes):
        obs = env.reset()
        nes = env._nes
        start_score = env._read_score()
        steps = 0
        for _ in range(max_steps):
            with torch.no_grad():
                logits, _ = net({"ram": torch.from_numpy(obs["ram"]).unsqueeze(0),
                                 "img": torch.from_numpy(obs["img"]).unsqueeze(0)})
            obs, _, done, info = env.step(int(logits.argmax()))
            steps += 1
            if done:
                break
        xs.append(info["x_progress"])
        lens.append(steps)
        axes.append(int(info["weapon"] > 0))
        kills += int(env._read_score() - start_score)
    return {"x": float(np.mean(xs)), "len": float(np.mean(lens)),
            "axe": float(np.mean(axes)), "score": kills,
            "xs": xs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=6)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, 8, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    print(f"checkpoint {a.checkpoint} ({ck['steps']} steps)\n")
    for flag in (False, True):
        r = run(net, flag, a.episodes)
        tag = "WITH axe at reset   " if flag else "normal game (no axe)"
        print(f"{tag}: mean x_progress {r['x']:7.1f} px | "
              f"mean episode {r['len']:6.1f} steps | "
              f"holding axe at end {r['axe']:.0%} | score {r['score']}")
        print(f"{'':21}  per-episode x: "
              + " ".join(f"{v:.0f}" for v in r["xs"]))


if __name__ == "__main__":
    main()
