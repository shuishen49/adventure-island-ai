"""Can the player throw the axe while airborne?

Until 2026-10-13 the action set had no "jump + throw" combination, so the
policy could only choose between jumping over a snail and shooting it.
This checks that the game actually accepts B while in the air.

Usage:  python tools/check_air_throw.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cynes import NES, NES_INPUT_A, NES_INPUT_B  # noqa: E402
import ram_map as R  # noqa: E402
from env import ROM  # noqa: E402
from visualize import boot  # noqa: E402


def shots(nes) -> int:
    return sum(1 for s in range(1, 16)
               if nes[R.OBJ.id + R.OBJ.STRIDE * s] == R.ID_PROJECTILE)


def main() -> None:
    nes = NES(ROM)
    boot(nes)
    nes[R.WEAPON] = 1                       # arm the axe
    print("ground y =", round(R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)), 1))

    # jump
    nes.controller = NES_INPUT_A
    for _ in range(12):
        nes.step()
    y_air = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    print("after 12 frames of A: y =", round(y_air, 1),
          "(smaller = higher up)")

    # throw while in the air
    nes.controller = NES_INPUT_A | NES_INPUT_B
    seen = 0
    for i in range(20):
        nes.step()
        y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        n = shots(nes)
        if n:
            seen += 1
        if i < 8:
            print(f"  air frame {i}: y={y:6.1f}  shots={n}")
    nes.controller = 0
    print(f"\nframes with a live axe in the air: {seen}/20")
    print("AIR THROW WORKS" if seen else "AIR THROW DOES NOT WORK")
    del nes


if __name__ == "__main__":
    main()
