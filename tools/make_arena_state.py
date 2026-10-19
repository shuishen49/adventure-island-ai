"""Save a state AT the 1-4 boss arena (progress 3592 px) for "boss practice".

Why: on 1-4 the scroll stops at 3592 px and the boss fight happens on a
fixed screen.  The agent currently dies around 3350 px (tools/
diag_killer_id.py: 12/12 deaths, best 3395), so it NEVER sees the arena --
which means the new boss-damage reward ($0011, see env.BOSS_HIT_GAIN) can
never fire and the fight can never be learned.

Fix: a curriculum entry that starts the episode already inside the arena.
Drive there with the usual invulnerability cheat (freeze $0072 = 0 and pin
$0076 = 10 so we cannot die or starve on the way), then snapshot the NES.

    python tools/make_arena_state.py --target 3592 --out saves/stages/stage_1-4arena.state

The file name must look like "<area>-<round>" because train.py parses the
stage id out of it; a snapshot of 1-4 (the game has no 1-5); it only ever holds this
arena snapshot (its one-hot bit is pool index 4, unused by real stages).
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
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,      # noqa: E402
                 NES_INPUT_A, NES_INPUT_B)

ROM_DIR = os.path.join(_ROOT, "saves", "stages")


def drive_to(env, target: float, max_frames: int = 60 * 60 * 6,
             throw: bool = True):
    """Run right (with hops) under an invulnerability cheat until target.

    throw=False is important when the target is right in front of the boss:
    a stray axe would damage it and the snapshot would not start at full
    HP ($0011 = 8).
    """
    nes = env._nes
    cam = env._cam
    best = 0.0
    stall = 0
    for f in range(max_frames):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        # run right, hop every ~24 frames to clear the small ledges
        ctl = NES_INPUT_RIGHT | (NES_INPUT_A if f % 24 < 4 else 0)
        if throw and f % 40 == 0:
            ctl |= NES_INPUT_B
        nes.controller = ctl
        nes.step()
        if f % 4 == 3:
            env._push_frame(np.asarray(nes.step(), dtype=np.uint8))
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        if best >= target:
            print(f"  到达 {best:.0f} px (frame {f})", flush=True)
            return best, True
        if stall > 45 * 60:
            print(f"  卡住于 {best:.0f} px (frame {f})", flush=True)
            return best, False
    print(f"  超时, 停在 {best:.0f} px", flush=True)
    return best, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=3592.0)
    ap.add_argument("--out", default=os.path.join(ROM_DIR, "stage_1-4arena.state"))
    ap.add_argument("--no-throw", action="store_true",
                    help="do not throw axes on the way (keeps boss HP full)")
    ap.add_argument("--from-state", default=None,
                    help="start from this save instead of booting round 3")
    args = ap.parse_args()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3,
                             start_state=args.from_state)
    env.reset(seed=0)
    nes = env._nes
    print(f"起点 progress = {env._cam.progress(nes):.0f} px  "
          f"$0011 = {int(nes[0x0011])}")

    best, ok = drive_to(env, args.target, throw=not args.no_throw)
    if not ok:
        print("!! 没走到 boss 场地, 不保存")
        return

    # let the arena settle: a few frames with no input and no cheat, so the
    # snapshot is a normal gameplay frame
    nes.controller = 0
    for _ in range(60):
        nes.step()
    nes[R.STAMINA] = 10
    nes[R.DEATH] = 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    state = nes.save()
    with open(args.out, "wb") as fh:
        fh.write(state)
    print(f"已保存 {args.out} ({len(state)} bytes)")

    # --- verify by reloading -----------------------------------------
    env2 = AdventureIslandEnv(start_with_axe=True, start_round=3,
                              stage_index=3, start_state=args.out)
    env2.reset(seed=0)
    n2 = env2._nes
    print(f"重载后 progress = {env2._cam.progress(n2):.0f} px  "
          f"$0011 = {int(n2[0x0011])}  "
          f"stamina = {int(n2[R.STAMINA])}  death = {int(n2[R.DEATH])}  "
          f"lives = {int(n2[R.LIVES])}")
    ids = [(s, int(n2[R.OBJ.id + R.OBJ.STRIDE * s]),
            round(R.px(R.read_obj_world_x(n2, s)), 1),
            round(R.px(R.read_obj_world_y(n2, s)), 1))
           for s in range(0, 16)
           if int(n2[R.OBJ.id + R.OBJ.STRIDE * s])]
    print("场上物体:", ids)


if __name__ == "__main__":
    main()
