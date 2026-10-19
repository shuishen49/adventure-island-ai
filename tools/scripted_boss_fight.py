#!/usr/bin/env python
"""Scripted boss fight: run in, hold the range, jump-throw on a cadence.

Used to answer ONE question that a learned policy cannot answer for us:
is a given snapshot *winnable at all*?  If this bot kills the boss from
saves/stages/stage_1-4gate.state but the policy scores 0 hits there, the
snapshot is fine and the policy simply has not learned that arrival state.

usage:
  python tools/scripted_boss_fight.py --state saves/stages/stage_1-4gate.state
  python tools/scripted_boss_fight.py --state saves/stages/stage_1-4boss.state
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                          # noqa: E402
from env import AdventureIslandEnv                           # noqa: E402

# ACTIONS index (see env.ACTIONS)
NOOP, RIGHT, LEFT, JUMP, RJ, LJ, THROW, RTHROW, JTHROW, RJTHROW = range(10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--far", type=int, default=95,
                    help="run right while the head is further than this")
    ap.add_argument("--near", type=int, default=38,
                    help="back off while the head is closer than this")
    ap.add_argument("--period", type=int, default=24)
    args = ap.parse_args()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=args.stage_idx,
                             start_state=args.state)
    o = env.reset(seed=1)
    nes = env._nes
    ID, XL = R.OBJ.id, R.OBJ.x_lo

    def head_dx():
        for s in range(1, 16):
            if int(nes[ID + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID:
                return int(nes[XL + R.OBJ.STRIDE * s]) - int(nes[XL])
        return None

    killed = False
    throws = 0
    log = []
    for n in range(args.steps):
        dx = head_dx()
        if dx is None:
            a = RIGHT
        elif dx > args.far:
            a = RIGHT
        elif dx < args.near:
            a = LEFT
        else:
            ph = n % args.period
            if ph == 0:
                a = RJ                      # running jump, carry speed
            elif ph in (6, 10, 14):
                a = RJTHROW                 # throw at the top of the arc
                throws += 1
            elif ph < 6:
                a = RJ
            else:
                a = NOOP
        o, _, done, info = env.step(a)
        if info.get("boss_killed"):
            killed = True
            log.append(f"n={n}: BOSS KILLED (hits={env._boss_hit_count()})")
            break
        if done:
            log.append(f"n={n}: episode over (death={int(nes[R.DEATH])})")
            break
        if n % 40 == 0:
            log.append(f"n={n:<4} dx={head_dx()} y={int(nes[R.OBJ.y_lo])} "
                       f"stam={int(nes[R.STAMINA])} "
                       f"hits={env._boss_hit_count()} act={a}")

    print(f"--- {os.path.basename(args.state)} ---")
    for line in log:
        print("  " + line)
    print(f"  killed={killed} throws={throws} "
          f"hits={env._boss_hit_count()} steps={n}")
    env.close()


if __name__ == "__main__":
    main()
