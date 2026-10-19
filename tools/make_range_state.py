#!/usr/bin/env python
"""Walk Higgins backwards/forwards inside a boss state until the head sits
at a chosen distance, then dump the frame.

Purpose: separate "the policy cannot fight here" from "the policy only
fights at ONE distance".  stage_1-4boss.state starts with the head 79 px
ahead and the policy scores 8/8 hits there; stage_1-4gate.state starts with
the head only ~55 px ahead and the policy scores 0 and just backs away.
Same arena, same boss, different range -- so shift the range and re-test.

usage:
  python tools/make_range_state.py --from saves/stages/stage_1-4gate.state \
      --out saves/stages/stage_1-4gate79.state --target-dx 79
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES                                            # noqa: E402
import ram_map as R                                              # noqa: E402
from env import (ROM, NES_INPUT_LEFT, NES_INPUT_RIGHT,           # noqa: E402
                 load_state_file)

ID, XL, YL = R.OBJ.id, R.OBJ.x_lo, R.OBJ.y_lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-dx", type=int, default=79)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--ground-y-min", type=int, default=138)
    ap.add_argument("--ground-y-max", type=int, default=148)
    args = ap.parse_args()

    nes = NES(ROM)
    nes.load(load_state_file(args.src))

    def head_dx():
        for s in range(1, 16):
            if int(nes[ID + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID:
                return (s, int(nes[XL + R.OBJ.STRIDE * s]) - int(nes[XL]))
        return (None, None)

    slot, dx0 = head_dx()
    if slot is None:
        print("no boss head in that state", file=sys.stderr)
        return
    print(f"start dx={dx0} (head slot {slot}), player x={int(nes[XL])}")

    want_back = dx0 < args.target_dx
    ctrl = NES_INPUT_LEFT if want_back else NES_INPUT_RIGHT
    best = None
    for f in range(args.max_frames):
        nes[R.STAMINA] = 10                 # do not let the script starve
        nes[R.DEATH] = 0
        nes.controller = ctrl
        nes.step()
        nes.controller = 0
        _, dx = head_dx()
        py = int(nes[YL])
        if dx is None:
            print(f"  frame {f}: boss gone (state unusable)")
            break
        if int(nes[R.DEATH]):
            print(f"  frame {f}: died while shifting")
            break
        ok = (abs(dx - args.target_dx) <= 3 and args.ground_y_min
              <= py <= args.ground_y_max)
        if ok:
            best = (dx, py, int(nes[XL]), bytes(nes.save()))
            break
        if f % 50 == 0:
            print(f"  frame {f}: dx={dx} y={py} playerx={int(nes[XL])}")

    if best is None:
        print("never reached the target range; try another --target-dx")
        return
    dx, py, px, blob = best
    with open(args.out, "wb") as fh:
        fh.write(blob)
    print(f"saved {args.out}: dx={dx}, player x={px} y={py} "
          f"({len(blob)} bytes)")


if __name__ == "__main__":
    main()
