"""How long is 1-4, really?

Under real invincibility the 14c policy scrolls 1-4 out past 25000 px
without $007A ever firing, while 1-3 clears at about 6600 px.  Before we
believe "1-4 is four times longer than 1-3", find the game's OWN level
pointer: some RAM byte must advance with the scroll (screen counter /
level data offset).  Scan every byte, keep the ones that track progress
monotonically, and report their range -- that range is the level length
in whatever unit the game uses.

    python tools/probe_14_length.py
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
from env import AdventureIslandEnv, N_ACTIONS, OBS_DIM     # noqa: E402
from train import ActorCritic                              # noqa: E402

N_RAM = 2048


def force_fairy(nes) -> None:
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 199:
            return
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 0:
            nes[R.OBJ.id + off] = 199
            for f in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr,
                      R.OBJ.y_lo, R.OBJ.y_fr):
                nes[f + off] = nes[f]
            return


def snap(nes) -> np.ndarray:
    return np.frombuffer(bytes(nes[a] for a in range(N_RAM)),
                         dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ppo_14c/ppo_final.pt")
    ap.add_argument("--round", type=int, default=3)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--every", type=int, default=150)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.round,
                             stage_index=args.stage)
    o = env.reset(seed=0)
    nes = env._nes

    snaps, progs = [], []
    for n in range(args.steps):
        force_fairy(nes)
        nes[R.STAMINA] = 10
        nes[R.LIVES] = 9
        with torch.no_grad():
            a = int(net({"ram": torch.from_numpy(o["ram"][None]),
                         "img": torch.from_numpy(o["img"][None])}
                        )[0].argmax(-1)[0])
        o, r, done, info = env.step(a)
        if n % args.every == 0:
            snaps.append(snap(nes))
            progs.append(float(info.get("best_progress", 0.0)))

    snaps = np.stack(snaps)
    progs = np.array(progs)
    print(f"{len(progs)} samples, progress {progs[0]:.0f} -> "
          f"{progs[-1]:.0f}")

    # rank bytes by how well they track progress (Spearman-ish)
    order = np.argsort(progs)
    pr = progs[order]
    monotone = []
    for a in range(N_RAM):
        v = snaps[:, a].astype(np.float64)[order]
        if v.std() < 1e-9:
            continue
        pv = np.argsort(np.argsort(v)).astype(np.float64)
        pp = np.argsort(np.argsort(pr)).astype(np.float64)
        if pv.std() < 1e-9:
            continue
        rho = float(np.corrcoef(pv, pp)[0, 1])
        if rho > 0.97:
            monotone.append((rho, a, int(v.min()), int(v.max())))
    monotone.sort(reverse=True)
    print(f"\nbytes tracking progress (rho>0.97): {len(monotone)}")
    for rho, a, lo, hi in monotone[:25]:
        print(f"  ${a:04X}  rho={rho:.4f}  range {lo} -> {hi}")


if __name__ == "__main__":
    main()
