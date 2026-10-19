#!/usr/bin/env python
"""Sanity check for a pre-arena snapshot: does the policy walk in and fight?

usage:
  python tools/check_approach_state.py --checkpoint runs/ppo_14l/ppo_X.pt \
      --state saves/stages/stage_1-4approach.state --episodes 3 --steps 1200
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R                                          # noqa: E402
from env import AdventureIslandEnv, N_ACTIONS, OBS_DIM       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()

    from train import ActorCritic
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    ID, XL = R.OBJ.id, R.OBJ.x_lo
    for ep in range(args.episodes):
        env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                                 stage_index=args.stage_idx,
                                 start_state=args.state)
        o = env.reset(seed=500 + ep)
        nes = env._nes
        seen = None
        hits = 0
        dxs = []
        for n in range(args.steps):
            with torch.no_grad():
                a = int(net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)}
                            )[0].argmax())
            o, _, done, info = env.step(a)
            slots = [s for s in range(1, 16)
                     if int(nes[ID + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID]
            if slots:
                if seen is None:
                    seen = n
                dxs.append(int(nes[XL + R.OBJ.STRIDE * slots[0]])
                           - int(nes[XL]))
                hits = env._boss_hit_count()
            if done:
                break
        print(f"ep{ep}: boss first seen at step {seen}, frames on screen "
              f"{len(dxs)}, dx first/mid/last = "
              f"{dxs[0] if dxs else '-'}/"
              f"{dxs[len(dxs)//2] if dxs else '-'}/"
              f"{dxs[-1] if dxs else '-'}, hits={hits}, "
              f"progress={info.get('best_progress')}, "
              f"done={done} step={n}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
