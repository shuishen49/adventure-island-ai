"""Which object id is the 1-4 boss, and where does it first appear?

probe_boss_locate.py showed the HUD "bar" is a constant 11-block element
(it is 11 from frame 0!), so it cannot locate the boss.  What did change
late in the stage was $0011 (8 -> 6) around progress 4300..4900, and the
object list there contained **id 59** (two slots, y = 168 / 184, i.e. on
the ground) and later id 198 + 67 -- the same id 59 probe_boss_v3c called
"the boss" after teleporting next to it.

This scan walks the stage with the invulnerability cheat and NO throwing,
recording for every object id the first progress value at which it shows
up.  Ids that only appear in the last stretch are the boss candidates.

    python tools/probe_boss_id.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,      # noqa: E402
                 NES_INPUT_A)


def main() -> None:
    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam

    first: dict[int, float] = {}
    last: dict[int, float] = {}
    best, stall = 0.0, 0
    hp_log = []
    prev_hp = int(nes[0x0011])

    for f in range(60 * 60 * 10):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if f % 26 < 4 else 0)
        nes.step()
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        for s in range(1, 16):
            oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
            if oid == 0:
                continue
            first.setdefault(oid, p)
            last[oid] = p
        hp = int(nes[0x0011])
        if hp != prev_hp:
            hp_log.append((f, best, prev_hp, hp))
            prev_hp = hp
        if stall > 45 * 60:
            print(f"卡住 {best:.0f} px @ f={f}", flush=True)
            break

    print(f"最远 {best:.0f} px\n")
    print("物体 id -> 首次出现 progress / 最后出现 progress:")
    for oid in sorted(first, key=lambda o: first[o]):
        name = R.OBJ_KINDS.get(oid, ("?", (0, 0, 0)))[0]
        print(f"  id{oid:<4} {name:<10} 首次 {first[oid]:7.1f}  "
              f"末次 {last[oid]:7.1f}"
              + ("   <-- 只在后半段出现" if first[oid] > best * 0.7 else ""))

    print("\n$0011 变化 (frame, 当时的最远 progress, old -> new):")
    for (f, p, a, b) in hp_log:
        print(f"  f={f:6d}  best={p:7.1f}  {a} -> {b}")


if __name__ == "__main__":
    main()
