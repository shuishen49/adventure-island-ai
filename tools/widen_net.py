"""Widen a trained checkpoint for the 32-stage one-hot observation.

The observation grows from 37 to 37 + N_STAGES(32) numbers when the
multi-stage curriculum is switched on.  You CANNOT just pass the old
checkpoint to ``--init-from``: that path inherits only tensors whose
SHAPE still matches, so ``fuse.0.weight`` ([128, n_flat+37] -> [128,
n_flat+69]) would be dropped and randomly re-initialised -- which throws
away everything the model knows, including stage 1-1.

Instead: copy every matching tensor, then rebuild the fusion matrix as

    [ old weights | zeros(128, 32) ]

New columns all zero => the stage one-hot contributes exactly 0 to the
pre-activation, so the widened network is BIT-IDENTICAL to the old one
until training moves those weights (plan.md phase 2).

usage:
    python tools/widen_net.py runs/ppo_v10/ppo_final.pt \
        runs/ppo_v10_wide/ppo_init.pt
    python tools/widen_net.py --check OLD NEW     # verify equivalence
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

from env import (IMG_H, IMG_STACK, IMG_W, N_ACTIONS, OBS_DIM,  # noqa: E402
                 N_STAGES, AdventureIslandEnv)
from train import ActorCritic  # noqa: E402

OLD_DIM = OBS_DIM - N_STAGES       # what the source checkpoint was built on


def widen(src_path: str, dst_path: str) -> None:
    ck = torch.load(src_path, map_location="cpu", weights_only=False)
    old = ck["model"]

    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    new = net.state_dict()

    inherited, zeroed = [], []
    for k, v in new.items():
        if k not in old:
            continue
        if old[k].shape == v.shape:
            new[k] = old[k].clone()
            inherited.append(k)
        elif k == "fuse.0.weight" and old[k].shape[0] == v.shape[0] \
                and v.shape[1] - old[k].shape[1] == N_STAGES:
            # concat zeros on the RIGHT: the one-hot tail is appended last
            # in env._build_obs, so the kept columns still line up with the
            # original features.
            new[k] = torch.cat(
                [old[k], torch.zeros(old[k].shape[0], N_STAGES)], dim=1)
            zeroed.append(k)
        else:
            raise SystemExit(
                f"cannot widen {k}: {tuple(old[k].shape)} -> {tuple(v.shape)}")

    net.load_state_dict(new)
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "steps": 0,
                "args": ck.get("args", {}),
                "widened_from": src_path}, dst_path)

    print(f"wrote {dst_path}")
    print(f"  obs_dim {OLD_DIM} -> {OBS_DIM}")
    print(f"  inherited unchanged: {len(inherited)} tensors")
    print(f"  widened with zeros : {zeroed}")
    print(f"  source: {src_path} ({ck.get('steps', '?')} steps)")


def check(src_path: str, dst_path: str, episodes: int = 3) -> None:
    """The widened net must behave EXACTLY like the old one on stage 1-1.

    plan.md phase 2 acceptance: same farthest progress, same death count.
    """
    old_ck = torch.load(src_path, map_location="cpu", weights_only=False)
    old_net = ActorCritic(OLD_DIM, N_ACTIONS, 128)
    old_net.load_state_dict(old_ck["model"])
    old_net.eval()

    new_ck = torch.load(dst_path, map_location="cpu", weights_only=False)
    new_net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    new_net.load_state_dict(new_ck["model"])
    new_net.eval()

    env = AdventureIslandEnv(seed=0)
    obs = env.reset(seed=0)

    def pick(net, dim):
        x = {"img": torch.as_tensor(obs["img"][None]),
             "ram": torch.as_tensor(obs["ram"][None][:, :dim],
                                    dtype=torch.float32)}
        with torch.no_grad():
            return int(net(x)[0].argmax())

    a_old = pick(old_net, OLD_DIM)
    a_new = pick(new_net, OBS_DIM)
    print(f"first action on 1-1: old={a_old} new={a_new} "
          f"{'OK' if a_old == a_new else 'MISMATCH'}")

    # logits must match to floating point, not just the argmax
    with torch.no_grad():
        lo = old_net({"img": torch.as_tensor(obs["img"][None]),
                      "ram": torch.as_tensor(obs["ram"][None][:, :OLD_DIM],
                                             dtype=torch.float32)})[0]
        ln = new_net({"img": torch.as_tensor(obs["img"][None]),
                      "ram": torch.as_tensor(obs["ram"][None],
                                             dtype=torch.float32)})[0]
    print(f"max |logit diff| = {(lo - ln).abs().max().item():.3e}")

    # and the one-hot must really be inert: zeroing it changes nothing
    z = obs["ram"].copy()
    z[-N_STAGES:] = 0.0
    with torch.no_grad():
        lz = new_net({"img": torch.as_tensor(obs["img"][None]),
                      "ram": torch.as_tensor(z[None], dtype=torch.float32)})[0]
    print(f"one-hot zeroed -> max |logit diff| = "
          f"{(ln - lz).abs().max().item():.3e} (must be 0)")

    # full rollouts on stage 1-1
    for net, dim, tag in ((old_net, OLD_DIM, "old "),
                          (new_net, OBS_DIM, "new ")):
        farthest = []
        for ep in range(episodes):
            o = env.reset(seed=ep)
            best, steps, done = 0.0, 0, False
            while not done and steps < 1200:
                x = {"img": torch.as_tensor(o["img"][None]),
                     "ram": torch.as_tensor(o["ram"][None][:, :dim],
                                            dtype=torch.float32)}
                with torch.no_grad():
                    a = int(net(x)[0].argmax())
                o, r, done, info = env.step(a)
                best = max(best, info["x_progress"])
                steps += 1
            farthest.append(best)
        print(f"  {tag}1-1 farthest px per episode: "
              + " ".join(f"{v:.0f}" for v in farthest))
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="runs/ppo_v10/ppo_final.pt")
    ap.add_argument("dst", nargs="?", default="runs/ppo_v10_wide/ppo_init.pt")
    ap.add_argument("--check", action="store_true",
                    help="do not write anything, just verify src vs dst")
    args = ap.parse_args()
    if args.check:
        check(args.src, args.dst)
    else:
        widen(args.src, args.dst)
        print()
        check(args.src, args.dst)


if __name__ == "__main__":
    main()
