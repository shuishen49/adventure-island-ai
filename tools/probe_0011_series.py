"""Is $0011 the boss HP byte?  Watch it across whole 1-4 episodes.

Evidence so far (2026-10-18):
  * every arena entry log says "progress=3592  $0011=8"
  * probe_boss_hp2: over 250 RAM snapshots inside the fight $0011 went
    8 -> 2 with 6 decreases and 0 increases (uniq 2..8)
  * probe_boss_v3c: 8,7,6,5,4,3,2,1 while axes were connecting, then the
    boss object changed and the bar reset

That is already a strong signal, but before we pay the agent for a drop we
must know it does NOT move anywhere else in the stage -- a byte that ticks
down on its own would hand out free reward and wreck the policy.

    python tools/probe_0011_series.py --checkpoint runs/ppo_14c/ppo_*.pt \
        --episodes 4 --greedy
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

from env import AdventureIslandEnv, OBS_DIM, N_ACTIONS, MAX_STEPS  # noqa: E402
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "runs", "ppo_14", "ppo_final.pt")
ADDR = 0x0011


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--start-round", type=int, default=3)
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--addr", default="0x0011")
    args = ap.parse_args()
    addr = int(args.addr, 0)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.start_round,
                             stage_index=args.stage_idx,
                             clear_ends_episode=True)

    drops_all = []
    for e in range(args.episodes):
        o = env.reset(seed=4000 + e)
        done, n = False, 0
        series, prog = [], []
        info = {}
        while not done and n < MAX_STEPS:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax()) if args.greedy else int(
                torch.distributions.Categorical(logits=lg[0]).sample())
            o, r, done, info = env.step(a)
            n += 1
            series.append(int(env._nes[addr]))
            prog.append(float(info.get("best_progress", 0.0)))

        s = np.array(series)
        p = np.array(prog)
        downs = np.where(np.diff(s) < 0)[0]
        ups = np.where(np.diff(s) > 0)[0]
        print(f"  ep{e + 1:2d}: 最好 {p.max():7.1f} px  {n:4d} 步  "
              f"${addr:04X} uniq={sorted(set(s.tolist()))}  "
              f"下降 {len(downs)} 次  上升 {len(ups)} 次")
        for d in downs:
            drops_all.append((e + 1, int(p[d]), int(s[d]), int(s[d + 1])))
            print(f"       下降 @ 步{d:4d}  progress={p[d]:7.1f}  "
                  f"{s[d]} -> {s[d + 1]}")
        for u in ups[:5]:
            print(f"       上升 @ 步{u:4d}  progress={p[u]:7.1f}  "
                  f"{s[u]} -> {s[u + 1]}")

    print()
    if not drops_all:
        print("整个 1-4 里 $%04X 一次都没下降过 (agent 还没打到 boss)" % addr)
    else:
        ps = [d[1] for d in drops_all]
        print(f"$%04X 共下降 {len(drops_all)} 次, 发生在 progress = " % addr
              + ", ".join(f"{x}" for x in ps))
        print("若全部 >= 3400 (boss 场地 3592 附近) 即可安全用作 boss 扣血奖励")


if __name__ == "__main__":
    main()
