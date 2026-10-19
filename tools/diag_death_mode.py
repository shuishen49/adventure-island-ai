"""Why does the runner die?  Fall / starve / contact -- for ANY stage.

tools/diag_killer_id.py only tells you *which object* was touching the
player.  On 2-1 every death frame had nothing but id64 (the player), so
the killer had to be the terrain or the stamina clock.  This script prints
the last frames before death: progress, player y, stamina, and the death
flag, then classifies each death.

    python tools/diag_death_mode.py CKPT --start-round 4 --stage-idx 4 \
        --episodes 6 [--greedy]
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
from env import (ACTIONS, OBS_DIM, AdventureIslandEnv)     # noqa: E402
from train import ActorCritic                              # noqa: E402


def load(path):
    net = ActorCritic(OBS_DIM, len(ACTIONS))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    own = net.state_dict()
    keep = {k: v for k, v in ck["model"].items()
            if k in own and own[k].shape == v.shape}
    net.load_state_dict(keep, strict=False)
    net.eval()
    return net


def classify(trace):
    """trace: list of (progress, y, stamina) for the last frames."""
    ys = [t[1] for t in trace]
    st = [t[2] for t in trace]
    falling = len(ys) >= 4 and (ys[-1] - max(ys[:3])) > 24
    starved = min(st) <= 1
    if starved and not falling:
        return "STARVE"
    if falling:
        return "FALL"
    return "CONTACT/OTHER"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--start-round", type=int, default=4)
    ap.add_argument("--stage-idx", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--tail", type=int, default=12)
    args = ap.parse_args()

    net = load(args.checkpoint)
    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.start_round,
                             stage_index=args.stage_idx,
                             clear_ends_episode=True)
    verdicts = {}
    for ep in range(args.episodes):
        obs = env.reset(seed=1000 + ep)
        nes, cam = env._nes, env._cam
        trace, done = [], False
        while not done:
            batch = {"ram": torch.as_tensor(obs["ram"], dtype=torch.float32)
                              .unsqueeze(0),
                     "img": torch.as_tensor(obs["img"], dtype=torch.float32)
                              .unsqueeze(0)}
            with torch.no_grad():
                logits, _ = net(batch)
            a = int(logits.argmax(-1)[0]) if args.greedy else \
                int(torch.distributions.Categorical(logits=logits).sample()[0])
            obs, r, done, info = env.step(a)
            act = ACTIONS[a][1] if a < len(ACTIONS) else str(a)
            trace.append((cam.progress(nes),
                          R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)),
                          int(nes[R.STAMINA]), act))
        dying = trace[-args.tail:]
        v = classify(trace[-40:])
        verdicts[v] = verdicts.get(v, 0) + 1
        print(f"ep{ep + 1}: {trace[-1][0]:.0f} px  step {len(trace)}  "
              f"-> {v}")
        print("   " + "  ".join(f"{p:.0f}/y{y:.0f}/s{int(s)}"
                                for p, y, s, _ in dying))
        print("   act: " + " ".join(a for _, _, _, a in dying))
    print("\nverdicts:", verdicts)


if __name__ == "__main__":
    main()
