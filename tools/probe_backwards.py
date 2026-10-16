"""The agent walks BACKWARDS -- why?

Replays the newest checkpoint greedily and logs, per decision step, the
action name and the world progress.  Then reports:

  * total forward vs backward movement
  * how often a LEFT action is chosen
  * the longest continuous backward stretch (with the actions in it)

Run:  python tools/probe_backwards.py runs/ppo_v3
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from env import ACTIONS, OBS_DIM, AdventureIslandEnv
from train import ActorCritic


def newest(folder: str) -> str:
    ck = sorted(glob.glob(os.path.join(folder, "ppo_*.pt")))
    return ck[-1] if ck else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="runs/ppo_v3")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=1200)
    a = ap.parse_args()

    path = (a.run_dir if a.run_dir.endswith(".pt") else newest(a.run_dir))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"checkpoint {path} ({ck['steps']} steps)\n")

    env = AdventureIslandEnv()
    counts = np.zeros(len(ACTIONS), dtype=int)
    for ep in range(a.episodes):
        obs = env.reset()
        prev = env._cam.progress(env._nes)
        fwd = back = 0.0
        traj = []           # (step, action_name, delta_progress)
        for s in range(a.max_steps):
            with torch.no_grad():
                logits, _ = net({
                    "ram": torch.from_numpy(obs["ram"]).unsqueeze(0),
                    "img": torch.from_numpy(obs["img"]).unsqueeze(0)})
            idx = int(logits.argmax())
            counts[idx] += 1
            obs, _, done, info = env.step(idx)
            now = env._cam.progress(env._nes)
            d = now - prev
            prev = now
            if d > 0:
                fwd += d
            else:
                back -= d
            traj.append((s, ACTIONS[idx][1], d))
            if done:
                break

        # longest continuous backward stretch
        best_len = best_start = 0
        cur_len = cur_start = 0
        cur_sum = best_sum = 0.0
        for i, (_, _, d) in enumerate(traj):
            if d < 0:
                if cur_len == 0:
                    cur_start, cur_sum = i, 0.0
                cur_len += 1
                cur_sum += d
                if cur_len > best_len:
                    best_len, best_start, best_sum = cur_len, cur_start, cur_sum
            else:
                cur_len = 0
        print(f"episode {ep}: {len(traj)} steps | forward {fwd:7.0f} px | "
              f"backward {back:7.0f} px | final x_progress "
              f"{info['x_progress']:.0f}")
        if best_len:
            seg = traj[best_start:best_start + best_len]
            print(f"   longest backward run: steps {best_start}.."
                  f"{best_start + best_len - 1} "
                  f"({best_sum:.0f} px lost), actions: "
                  + ", ".join(n for _, n, _ in seg[:25])
                  + (" ..." if best_len > 25 else ""))
        # sample of the trajectory every 40 steps
        print("   trace: " + " ".join(
            f"{s}:{n}{d:+.0f}" for s, n, d in traj[::40]))

    print("\naction mix (greedy):")
    for i, (_, name) in enumerate(ACTIONS):
        if counts[i]:
            print(f"  {name:12s} {counts[i]:6d}  "
                  f"{counts[i] / counts.sum():6.1%}")


if __name__ == "__main__":
    main()
