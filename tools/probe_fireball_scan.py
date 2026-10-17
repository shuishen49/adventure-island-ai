"""Where is the FIREBALL weapon type stored?

v4 proved `$006D` is only a boolean ("armed / not armed"): sweeping it
1..8 gave byte-identical shots (id 66, 139 px).  So the weapon TYPE must
live somewhere else.  This scans for it.

Method: drive the real env (the only harness that reliably spawns a
projectile -- a raw poke + 3-frame button press never does), force ONE
byte to a candidate value on every decision, throw ONCE, then stand still
and measure how far the shot gets before it expires.  The wiki says the
fireball "reaches farther" and "cuts through rocks", so a longer reach or
a different object id is the fingerprint.

    python tools/probe_fireball_scan.py [--lo 0x00] [--hi 0x100]
"""
from __future__ import annotations

import argparse
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
NOOP = 0
OBJ_ID = R.OBJ.id
STRIDE = R.OBJ.STRIDE
N_SLOTS = 16


def one_shot(env, nes, poke=None, watch=25):
    """Throw once, then idle; return (ids seen, max dx, frames alive)."""
    if poke is not None:
        addr, val = poke
        nes[addr] = val
    nes[R.WEAPON] = 1
    x0 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    env.step(THROW)
    ids, reach, life, pid = set(), -1e9, 0, None
    for _ in range(watch):
        if poke is not None:
            nes[poke[0]] = poke[1]
        env.step(NOOP)
        for s in range(N_SLOTS):
            oid = int(nes[OBJ_ID + STRIDE * s])
            if oid in (0, R.PLAYER_ID, 16, 17, 65, 28, 43, 69):
                continue
            ids.add(oid)
            dx = R.px(R.read_obj_world_x(nes, s)) - x0
            if dx > reach:
                reach, pid = dx, oid
            life += 1
    return sorted(ids), reach, life, pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=lambda s: int(s, 16), default=0x00)
    ap.add_argument("--hi", type=lambda s: int(s, 16), default=0x100)
    ap.add_argument("--vals", default="1,2,3")
    args = ap.parse_args()
    vals = [int(v) for v in args.vals.split(",")]

    env = AdventureIslandEnv(start_state=STATE, start_with_axe=True,
                             stage_index=0)
    nes = env._nes
    env.reset(seed=0)
    base_state = np.frombuffer(nes.save(), dtype=np.uint8).copy()

    def reload():
        nes.load(base_state)

    reload()
    b_ids, b_reach, b_life, b_pid = one_shot(env, nes)
    print(f"[baseline axe] ids {b_ids} id={b_pid} reach {b_reach:.1f} px "
          f"alive {b_life} slot-frames\n")

    print(f"scanning ${args.lo:04X}..${args.hi:04X} x vals {vals}")
    for addr in range(args.lo, args.hi):
        if addr == R.WEAPON:
            continue
        for v in vals:
            reload()
            try:
                ids, reach, life, pid = one_shot(env, nes, poke=(addr, v))
            except Exception as exc:      # pragma: no cover
                print(f"  ${addr:04X}={v}: emulator error {exc}")
                continue
            diff = (pid != b_pid) or (reach > b_reach + 8)
            if diff:
                print(f"  ${addr:04X}={v}: ids {ids} id={pid} "
                      f"reach {reach:.1f} px alive {life}   <-- DIFFERENT",
                      flush=True)
    print("done")
    env.close()


if __name__ == "__main__":
    main()
