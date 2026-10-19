"""Photograph a stretch of a stage while the policy plays it.

Used when the runner dies the same way every time (see
tools/diag_death_mode.py: on 2-1 the player's y goes 53 -> 240 in twelve
steps and dies, with stamina still at 10).  This walks the stage with the
policy and saves an annotated PNG every N steps once --from is reached,
so the terrain / gap can be inspected frame by frame.

    python tools/snap_fall.py CKPT --round 4 --stage-idx 4 \
        --from 4760 --to 4930 --every 3 --out outputs/fall21
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
from env import AdventureIslandEnv                         # noqa: E402
from train import ActorCritic                              # noqa: E402
from env import OBS_DIM, ACTIONS                           # noqa: E402
from snap_unknown import annotate                          # noqa: E402


def load(path):
    net = ActorCritic(OBS_DIM, len(ACTIONS))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    own = net.state_dict()
    keep = {k: v for k, v in ck["model"].items()
            if k in own and own[k].shape == v.shape}
    net.load_state_dict(keep, strict=False)
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--stage-idx", type=int, default=4)
    ap.add_argument("--from", dest="frm", type=float, default=4760.0)
    ap.add_argument("--to", type=float, default=4930.0)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "fall"))
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = load(args.checkpoint)
    env = AdventureIslandEnv(start_with_axe=True, start_round=args.round,
                             stage_index=args.stage_idx,
                             clear_ends_episode=True)
    n = 0
    for ep in range(args.episodes):
        obs = env.reset(seed=1000 + ep)
        nes, cam = env._nes, env._cam
        done = False
        step = 0
        while not done:
            batch = {"ram": torch.as_tensor(obs["ram"], dtype=torch.float32)
                              .unsqueeze(0),
                     "img": torch.as_tensor(obs["img"], dtype=torch.float32)
                              .unsqueeze(0)}
            with torch.no_grad():
                logits, _ = net(batch)
            a = int(logits.argmax(-1)[0])
            obs, r, done, info = env.step(a)
            step += 1
            p = cam.progress(nes)
            if args.frm <= p <= args.to and step % args.every == 0:
                py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
                img = annotate(nes.render_frame()
                               if hasattr(nes, "render_frame")
                               else env._last_frame, nes, args.scale)
                path = os.path.join(args.out, f"f{n:03d}_{p:.0f}px_y{py:.0f}"
                                              f"_s{int(nes[R.STAMINA])}.png")
                img.save(path)
                print(f"  {os.path.basename(path)}", flush=True)
                n += 1
            if p > args.to:
                break
    print(f"{n} shots -> {args.out}")


if __name__ == "__main__":
    main()
