"""Is there a FIREBALL weapon, and where is it in RAM?  (v4, env-based)

Game rules (Wikipedia / neoseeker FAQ / nesrocks' disassembly blog):
Higgins starts unarmed; the stone axe comes from an egg; **if he already
has the axe**, a hidden "power up egg" can trade it for **magical
fireballs** -- longer range, and the ONLY thing that can destroy rocks and
rolling boulders (they are invulnerable to the stone axe).

Memory editing alone is not enough here: a raw poke + button press turned
out to need the exact 4-frame cadence the env uses (a 3-frame press never
spawns the projectile).  So we drive the REAL environment and force
`$006D` to the value under test on every decision, then measure what flew
out: its object id, how far it got, and how long it lived.

    python tools/probe_fireball.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R  # noqa: E402
from env import ACTIONS, AdventureIslandEnv  # noqa: E402

STATE = os.path.join(_ROOT, "saves", "stages", "stage_1-1.state")
THROW = [i for i, a in enumerate(ACTIONS) if a[1] == "throw"][0]
OBJ_ID = R.OBJ.id
STRIDE = R.OBJ.STRIDE
N_SLOTS = 16


def probe(weapon_value: int, decisions: int = 10, stage: str = STATE):
    """Force $006D = weapon_value, throw repeatedly, measure the shot."""
    env = AdventureIslandEnv(start_state=stage, start_with_axe=True,
                             stage_index=0)
    env.reset(seed=0)
    nes = env._nes
    max_dx, life, pid, ids = -1e9, 0, None, set()
    try:
        for k in range(decisions):
            nes[R.WEAPON] = weapon_value          # the "cheat"
            x0 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
            env.step(THROW)
            for s in range(N_SLOTS):
                oid = int(nes[OBJ_ID + STRIDE * s])
                if oid in (0, R.PLAYER_ID):
                    continue
                ids.add(oid)
                dx = R.px(R.read_obj_world_x(nes, s)) - x0
                if oid == 66 or oid not in (16, 17, 65, 28, 43, 69):
                    if dx > max_dx:
                        max_dx, pid = dx, oid
                    life += 1
    finally:
        env.close()
    return {"ids": sorted(ids), "reach": max_dx, "pid": pid, "life": life}


def main():
    print("=== $006D swept 0..8 (stage 1-1, 10 throw-decisions each) ===")
    base = None
    for v in range(9):
        r = probe(v)
        print(f"  $006D={v}: ids {r['ids']}  shot id {r['pid']}  "
              f"reach {r['reach']:6.1f} px  seen {r['life']} slot-frames",
              flush=True)
        if v == 1:
            base = r
    print("\n(斧头基线 = $006D=1；如果某个值让射程明显变长或弹体 id 变了，那就是火焰弹)")


if __name__ == "__main__":
    main()
