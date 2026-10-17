"""Scan what each value of the round register $0038 actually loads.

The register is 0-based (stage 1-1 plays with round = 0).  Adventure Island
has 8 areas (islands) x 4 rounds = 32 stages, and $0037 holds the area, so
a correct warp should show:

    round 0..3   -> area 0   (island 1: stages 1-1 .. 1-4)
    round 4..7   -> area 1   (island 2: stages 2-1 .. 2-4)
    ...

That area bump is the hard evidence that we are really warping to a later
stage and not just loading a differently-seeded copy of stage 1-1.

usage:
    python tools/probe_rounds.py            # rounds 0..7
    python tools/probe_rounds.py --max 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402
from env import ROM, control_live, wait_for_picture  # noqa: E402

SHOT_DIR = os.path.join(_ROOT, "outputs", "probe_rounds")


def warp_to(round_value: int):
    """Boot and force the round register.  Returns (nes, info) or (None, e)."""
    nes = NES(ROM)
    nes.step(frames=300)                       # title screen
    nes.controller = NES_INPUT_START
    for _ in range(90):
        nes[R.ROUND] = round_value
        nes.step()
    nes.controller = 0
    for _ in range(600):
        if control_live(nes):
            break
        for _ in range(10):
            nes.controller = 0
            nes.step()
    else:
        del nes
        return None, "never got control"
    wait_for_picture(nes)
    if not control_live(nes):
        del nes
        return None, "lost control during fade-in"
    return nes, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=7,
                    help="highest round register value to probe")
    args = ap.parse_args()
    os.makedirs(SHOT_DIR, exist_ok=True)

    print(f"{'reg':>4} {'stage':>7} {'area':>5} {'round':>6} "
          f"{'ckpt':>5} {'spawn':>14} {'bright':>7} {'live':>5}")
    for v in range(args.max + 1):
        nes, err = warp_to(v)
        if nes is None:
            print(f"{v:>4} {'--':>7} {err}")
            continue
        arr = np.asarray(nes.step(), dtype=np.uint8).copy()
        Image.fromarray(arr, mode="RGB").resize(
            (512, 480), Image.NEAREST).save(
                os.path.join(SHOT_DIR, f"round_{v:02d}.png"))
        px = (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
              R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))
        area, rd = nes[R.AREA], nes[R.ROUND]
        # the game's own labels are 1-based: area+1, round+1
        print(f"{v:>4} {area + 1}-{rd + 1:>5} {area:>5} {rd:>6} "
              f"{nes[R.CHECKPOINT]:>5} "
              f"{f'({px[0]:.0f},{px[1]:.0f})':>14} "
              f"{arr.mean():>7.1f} "
              f"{'yes' if control_live(nes) else 'NO':>5}")
        del nes


if __name__ == "__main__":
    main()
