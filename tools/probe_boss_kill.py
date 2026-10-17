"""Can the 1-4 boss (slots id 198 + 67) be damaged at all?

The camera locks at ~2591 px for the fight, so progress rewards stop and
the ONLY signal PPO would get is CLEAR_GAIN when the boss finally dies.
Before spending hours of training on it we must know the fight is even
solvable with our 12-action set.

This probe parks the player at a few distances in front of the boss and
tries several throw rhythms (ground, jump, jump-while-moving), watching
for: the boss slots changing id / vanishing, the score moving, or
$007A (stage complete) firing.

    python tools/probe_boss_kill.py 1-4
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R
from env import ACTIONS, AdventureIslandEnv

OID, ST = R.OBJ.id, R.OBJ.STRIDE
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]
A_JT = [i for i, a in enumerate(ACTIONS) if a[1] == "jump+throw"][0]
A_RJT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump+throw"][0]
A_NOOP = [i for i, a in enumerate(ACTIONS) if a[1] == "noop"][0]

PATTERNS = {
    "ground-fast": lambda t: A_RT if t % 4 in (0,) else A_RIGHT,
    "ground-slow": lambda t: A_RT if t % 10 in (0, 1) else A_RIGHT,
    "jump-throw": lambda t: A_RJT if t % 9 in (0, 1, 2) else A_RIGHT,
    "stand-throw": lambda t: A_RT if t % 6 in (0, 1) else A_NOOP,
    "stand-jump": lambda t: (A_JT if t % 12 in (0, 1)
                             else (A_NOOP if t % 12 < 6 else A_RIGHT)),
}


def to_arena(nes, env) -> int:
    for t in range(3000):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        if any(int(nes[OID + ST * s]) in (198, 67) for s in range(16)):
            return t
        env.step(A_RJ if t % 26 in (0, 1) else (A_RT if t % 11 == 0 else A_RIGHT))
    return -1


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-4"
    area, rnd = (int(p) for p in name.split("-"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    state = os.path.join(root, "saves", "stages", f"stage_{name}.state")

    for tag, fn in PATTERNS.items():
        env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                                 stage_index=(area - 1) * 4 + (rnd - 1))
        env.reset(seed=1)
        nes = env._nes
        t0 = to_arena(nes, env)
        s0 = int(nes[R.SCORE])
        ids0 = [int(nes[OID + ST * s]) for s in range(16)]
        changes = 0
        prev = ids0
        killed = dead_at = -1
        sc_max = s0
        for t in range(1200):
            if int(nes[R.DEATH]) != 0:
                nes[R.DEATH] = 0
            nes[R.STAMINA] = 10
            nes[R.WEAPON] = 1
            env.step(fn(t))
            sc_max = max(sc_max, int(nes[R.SCORE]))
            cur = [int(nes[OID + ST * s]) for s in range(16)]
            if cur != prev:
                changes += 1
            prev = cur
            if int(nes[R.STAGE_COMPLETE]) != 0:
                killed, dead_at = 1, t
                break
            if not any(c in (198, 67) for c in cur):
                killed, dead_at = 2, t
                break
        print(f"{tag:12s} 到竞技场t={t0:4d}  结果="
              f"{'过关' if killed == 1 else ('boss消失' if killed == 2 else '无进展')}"
              f" @{dead_at if dead_at >= 0 else '-'}  "
              f"score {s0}->{sc_max}  槽位变化 {changes}/1200")
        sys.stdout.flush()
        env.close()


if __name__ == "__main__":
    main()
