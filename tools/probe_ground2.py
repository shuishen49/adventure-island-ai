"""Validate a terrain probe: can it SEE the pit before the player falls?

Rule under test: for a screen column, scan from the bottom up and report
the topmost brown (ground-coloured) pixel.  A pit should report "no ground"
(or a surface far below the normal one).
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot

OFFS = (-24, -8, 0, 8, 16, 24, 32, 40, 48, 64, 80)


def brown(frame):
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    return (r > 100) & (r > g) & (g >= b)


def profile(frame, cols):
    m = brown(frame)[:, cols]                 # (240, n)
    out = []
    for j in range(m.shape[1]):
        ys = np.flatnonzero(m[:, j])
        out.append(int(ys[0]) if ys.size else 240)
    return out


def main():
    nes = NES(ROM)
    boot(nes)
    nes.controller = NES_INPUT_RIGHT

    hist = []
    for i in range(700):
        frame = np.asarray(nes.step(), dtype=np.uint8)
        px = int(R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)))
        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        cols = [int(np.clip(px + o, 0, 255)) for o in OFFS]
        prof = profile(frame, cols)
        hist.append((i, py, prof))
        if len(hist) > 40:
            hist.pop(0)
        if py > 200 and len(hist) > 12:
            print(f"坠落发生在帧 {i}, 玩家屏幕 y={py:.0f}")
            print("  坠落前 12 帧的地形剖面 (数字=地面高度, 240=看不到地面)")
            print("        " + "  ".join(f"{o:+4d}" for o in OFFS))
            for fr, yy, pr in hist[-12:]:
                mark = "  <-- 坠落" if fr == i else ""
                print(f"  f{fr:4d} y{yy:6.1f} " +
                      "  ".join(f"{v:4d}" for v in pr) + mark)
            break
    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
