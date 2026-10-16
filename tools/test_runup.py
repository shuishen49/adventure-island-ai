"""Closed-loop test: back off, run up, and jump at a chosen distance.

If a well-timed running jump clears the obstacle, the task is solvable and
the agent just needs better exploration.  If nothing clears it, the
environment itself blocks all progress.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_A, NES_INPUT_LEFT, NES_INPUT_RIGHT  # noqa: E402
from cynes import NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = r"D:\source_code\cynes\冒险岛1.nes"


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def px_x(nes):
    return R.px(R.read_obj_world_x(nes, 0))


def trial(jump_at, jump_frames=12, back_to=20, total=900):
    """Go left to `back_to`, run right, jump when x crosses `jump_at`."""
    nes = NES(ROM)
    boot(nes)
    best = px_x(nes)

    # walk left to build up runway
    nes.controller = NES_INPUT_LEFT
    for _ in range(300):
        nes.step()
        if px_x(nes) <= back_to:
            break
    nes.controller = 0
    nes.step(frames=20)

    jumping, jt = False, 0
    for _ in range(total):
        x = px_x(nes)
        best = max(best, x)
        ctrl = NES_INPUT_RIGHT
        if not jumping and x >= jump_at:
            jumping, jt = True, jump_frames
        if jt > 0:
            ctrl |= NES_INPUT_A
            jt -= 1
        else:
            jumping = False
        nes.controller = ctrl
        nes.step()
    best = max(best, px_x(nes))
    del nes
    return best


print("runway + jump-at-distance trials (obstacle sits around x=105-120):")
for at in (40, 55, 70, 85, 95):
    b = trial(at)
    print(f"  jump_at={at:3d}px -> furthest x = {b:6.1f}  "
          f"{'CLEARED!' if b > 130 else 'still blocked'}")

print("\nvariants at jump_at=60:")
for jf in (6, 10, 16, 24):
    b = trial(60, jump_frames=jf)
    print(f"  jump_frames={jf:3d} -> furthest x = {b:6.1f}  "
          f"{'CLEARED!' if b > 130 else 'still blocked'}")
