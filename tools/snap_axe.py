"""Save annotated frames around the moment the axe is picked up.

Drives the game with a simple "chase the nearest item" controller (no
neural net needed) and writes two PNGs next to each other:
  outputs/axe_before.png   - the egg is on screen, AXE = no
  outputs/axe_after.png    - one second later, AXE = YES

Usage:  python tools/snap_axe.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cynes import NES, NES_INPUT_A, NES_INPUT_LEFT, NES_INPUT_RIGHT  # noqa: E402
import ram_map as R  # noqa: E402
from visualize import annotate, boot  # noqa: E402

ROM = os.environ.get(
    "AI_ROM",
    r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes",
)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")


def nearest_item(nes):
    p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
    p_y = R.read_obj_world_y(nes, R.PLAYER_SLOT)
    best = None
    for s in range(1, 15):
        if nes[R.OBJ.id + R.OBJ.STRIDE * s] not in R.ITEM_IDS:
            continue
        dx = R.px(R.read_obj_world_x(nes, s) - p_x)
        dy = R.px(R.read_obj_world_y(nes, s) - p_y)
        if abs(dx) > 200:
            continue
        if best is None or abs(dx) < abs(best[0]):
            best = (dx, dy)
    return best


def main() -> None:
    nes = NES(ROM)
    boot(nes)
    rng = np.random.default_rng(0)

    before = None
    for f in range(60 * 90):
        item = nearest_item(nes)
        if item is None:
            ctl = NES_INPUT_RIGHT if rng.random() < 0.7 else 0
            if rng.random() < 0.2:
                ctl |= NES_INPUT_A                          # jump
        else:
            dx, dy = item
            ctl = NES_INPUT_RIGHT if dx > 0 else NES_INPUT_LEFT
            if dy < -4:
                ctl |= NES_INPUT_A
        nes.controller = ctl
        frame = np.asarray(nes.step(), dtype=np.uint8)

        if nes[R.WEAPON] == 0:
            if before is None and nearest_item(nes) is not None:
                before = annotate(frame, nes, "chase ITEM")
        else:
            after = annotate(frame.copy(), nes, "GOT AXE")
            if before is not None:
                w = before.shape[1]
                both = np.concatenate([before, after], axis=1)
                Image.fromarray(both).resize((w * 4, before.shape[0] * 2),
                                             Image.NEAREST).save(
                    os.path.join(OUT, "axe_pickup.png"))
                print("saved outputs/axe_pickup.png")
            else:
                print("picked the axe up but no 'before' frame was kept")
            break
    else:
        print("never picked the axe up within 90 s of game time")
    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
