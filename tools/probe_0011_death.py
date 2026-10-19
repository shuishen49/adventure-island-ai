"""$0011: boss HP, or just a death counter?  Settle it with a clean test.

Two observations contradict each other:

  * probe_boss_v3c: teleported in front of the boss and spammed axes --
    $0011 went 8 -> 1 and the boss object changed.
  * tools/probe_14_end.py: a cheat runner that kept FALLING INTO PITS and
    respawning also made $0011 fall 8 -> 5, with no boss in sight.

So maybe $0011 simply counts deaths / remaining stock.  This test never
throws a single axe: stand still and starve (verified: $0076 reaches 0 and
the death flag fires on that exact frame).  If $0011 drops there, it is a
death counter and the "boss HP" reward would hand out +6 for DYING.

    python tools/probe_0011_death.py [--deaths 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import AdventureIslandEnv                         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deaths", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=60 * 60 * 5)
    args = ap.parse_args()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes

    prev = int(nes[0x0011])
    lives0 = int(nes[R.LIVES])
    deaths = 0
    print(f"起点 $0011={prev}  lives={lives0}  stamina={int(nes[R.STAMINA])}")
    for f in range(args.max_frames):
        nes.controller = 0            # stand still -> starve
        nes.step()
        hp = int(nes[0x0011])
        if hp != prev:
            print(f"  f={f:6d}  $0011 {prev} -> {hp}   "
                  f"stamina={int(nes[R.STAMINA])}  death={int(nes[R.DEATH])}"
                  f"  lives={int(nes[R.LIVES])}  camera={int(nes[R.CAMERA])}",
                  flush=True)
            prev = hp
            if hp < prev:
                pass
        if nes[R.DEATH] != 0 and nes[R.STAGE_COMPLETE] == 0:
            deaths += 1
            print(f"  f={f:6d}  >>> 死亡 #{deaths}  "
                  f"lives={int(nes[R.LIVES])}  $0011={int(nes[0x0011])}",
                  flush=True)
            if deaths >= args.deaths:
                break
            # let the respawn finish
            for _ in range(400):
                nes.controller = 0
                nes.step()
        if f % 600 == 0:
            print(f"  f={f:6d}  $0011={int(nes[0x0011])}  "
                  f"stamina={int(nes[R.STAMINA])}  lives={int(nes[R.LIVES])}",
                  flush=True)

    print(f"\n共死亡 {deaths} 次, $0011 终值 {int(nes[0x0011])}  "
          f"(起点 {prev})  lives {lives0} -> {int(nes[R.LIVES])}")


if __name__ == "__main__":
    main()
