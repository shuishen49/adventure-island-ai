"""A scripted boss-fight bot: keep the right distance, jump-throw forever,
and watch for ANY sign that the boss took damage or died.

Why: every short probe so far ended with "nothing changed".  The axe does
reach the head (it freezes at the head's lower edge around frame 31 in a
run-right + jump-throw script), yet no byte moves -- so the boss is simply
invulnerable in that state, and we have to find the state in which it is
not.  Instead of guessing, we fight for a long time and log every event:

  * head id changes / disappears   -> boss died or changed form
  * $007A stage complete
  * any byte in the "small counter" watch-list stepping down

    python tools/boss_fight_bot.py --frames 6000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import (ROM, NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT,  # noqa: E402
                 NES_INPUT_RIGHT, load_state_file)
from cynes import NES                                            # noqa: E402

ID, XL, YL = 0x0564, 0x0584, 0x05C4
HEAD, BODY = 14, 15
STAMINA, DEATH, STAGE_COMPLETE = 0x0076, 0x0072, 0x007A
WATCH = [0x0011, 0x0013, 0x0016, 0x0672, 0x0673, 0x0682, 0x0683,
         0x0692, 0x0693, 0x007A, 0x0039, 0x0038]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_bossface.state"))
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--keep", type=int, default=40,
                    help="try to hold the head this many px ahead")
    ap.add_argument("--jump-period", type=int, default=30)
    ap.add_argument("--throw-delay", type=int, default=12)
    args = ap.parse_args()

    nes = NES(ROM)
    nes.load(load_state_file(args.state))
    prev_w = {a: int(nes[a]) for a in WATCH}
    events = []
    head_alive = True
    for f in range(args.frames):
        nes[STAMINA] = 10
        nes[DEATH] = 0
        hx = int(nes[XL + HEAD])
        px = int(nes[XL])
        dx = hx - px
        btn = 0
        if dx > args.keep + 20:
            btn |= NES_INPUT_RIGHT
        elif dx < args.keep - 20:
            btn |= NES_INPUT_LEFT
        if f % args.jump_period == 0:
            btn |= NES_INPUT_A
        if f % args.jump_period == args.throw_delay:
            btn |= NES_INPUT_B
        nes.controller = btn
        nes.step()
        nes.controller = 0

        hid = int(nes[ID + HEAD])
        if hid != 198 and head_alive:
            head_alive = False
            events.append((f, f"HEAD id -> {hid}"))
        if int(nes[STAGE_COMPLETE]):
            events.append((f, f"STAGE_COMPLETE={int(nes[STAGE_COMPLETE])}"))
        for a in WATCH:
            v = int(nes[a])
            if v != prev_w[a]:
                if a in (0x0011, 0x007A, 0x0039):
                    events.append((f, f"0x{a:04X} {prev_w[a]} -> {v}"))
                prev_w[a] = v

    print(f"frames={args.frames}  head still 198: {head_alive}")
    print(f"{len(events)} events")
    for f, e in events[:60]:
        print(f"  f{f}: {e}")
    np.save(os.path.join(_ROOT, "outputs", "boss_bot_tail.npy"),
            np.array([[int(nes[a]) for a in range(0x800)]], dtype=np.uint8))


if __name__ == "__main__":
    main()
