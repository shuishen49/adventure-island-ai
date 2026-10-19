"""Find the byte that REALLY means "stage cleared".

$007A is what env.py calls STAGE_COMPLETE, but tools/probe_14_final.py
and this script's predecessor showed it can go to 1 while DEATH is 255 --
i.e. it may just be a flag inside the death routine.  Nothing on 1-4 has
ever made it fire.

So: run a model that genuinely clears a stage (1-2), snapshot the whole
2 KB of RAM every frame into a ring buffer, and when env.py reports
`cleared` diff the RAM against 90 frames earlier.  Whatever changed is
the real "stage end" machinery.

    python tools/probe_true_clear.py --checkpoint models/ppo_1-2_clear_8M.pt
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import (AdventureIslandEnv, OBS_DIM, N_ACTIONS, MAX_STEPS)  # noqa
from train import ActorCritic                                        # noqa

RAMP = " .:-=+*#%@"
N_RAM = 2048


def ascii_art(frame: np.ndarray, w: int = 76, h: int = 30) -> str:
    g = frame.astype(np.float32).mean(axis=2)
    H, W = g.shape
    ys = np.linspace(0, H, h + 1).astype(int)
    xs = np.linspace(0, W, w + 1).astype(int)
    return "\n".join(
        "".join(RAMP[min(len(RAMP) - 1,
                         int(g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
                             / 255.0 * len(RAMP)))]
                for j in range(w))
        for i in range(h))


def dump(nes):
    return np.frombuffer(
        bytes(nes[a] for a in range(N_RAM)), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/ppo_1-2_clear_8M.pt")
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--stage-idx", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.start_round,
                             stage_index=args.stage_idx,
                             clear_ends_episode=True)
    print(f"model={args.checkpoint} round={args.start_round} "
          f"stage={args.stage_idx}")

    for ep in range(args.episodes):
        o = env.reset(seed=1000 + ep)
        nes = env._nes
        ring = collections.deque(maxlen=200)
        n = 0
        done = False
        while not done and n < MAX_STEPS:
            b = {"ram": torch.from_numpy(o["ram"][None]),
                 "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                a = int(net(**b)[0].argmax(-1)[0])
            ring.append(dump(nes))
            o, r, done, info = env.step(a)
            n += 1
            if info.get("cleared"):
                print(f"\n=== episode {ep}: env says CLEARED at step {n} "
                      f"(progress {info.get('progress', 0):.0f}) ===",
                      flush=True)
                print(f"    DEATH={int(nes[0x0072])} LIVES="
                      f"{int(nes[0x003F])} $007A="
                      f"{int(nes[0x007A])}")
                old = ring[-90] if len(ring) >= 90 else ring[0]
                new = dump(nes)
                diff = np.nonzero(old != new)[0]
                print(f"    {len(diff)} RAM bytes changed vs 90 frames "
                      f"earlier:")
                for a2 in diff[:60]:
                    print(f"      ${a2:04X}: {int(old[a2])} -> "
                          f"{int(new[a2])}")
                print(ascii_art(np.asarray(nes.step(0), dtype=np.uint8)
                                if False else np.zeros((240, 256, 3),
                                                       np.uint8)))
                break
        else:
            print(f"episode {ep}: no clear "
                  f"(steps={n} prog={info.get('progress', 0):.0f} "
                  f"deaths={info.get('deaths', '?')})")


if __name__ == "__main__":
    main()
