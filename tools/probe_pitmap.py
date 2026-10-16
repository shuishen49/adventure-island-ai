"""Render an ASCII map of the terrain around the player's feet."""
import sys

import numpy as np

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot


def classify(p):
    r, g, b = (int(v) for v in p)
    if r < 30 and g < 30 and b < 30:
        return "#"            # black: pit / outline
    if g > 150 and r < 150:
        return "."            # green foliage / sky
    if r > 100 and r > g > b:
        return "G"            # brown ground
    if r > 200 and g > 180:
        return "o"            # bright (fruit, player, ...)
    return "-"


def amap(frame, x0, y0, w, h):
    lines = []
    for y in range(y0, min(y0 + h, 240)):
        row = "".join(classify(frame[y, x])
                      for x in range(max(0, x0), min(x0 + w, 256)))
        lines.append(row)
    return "\n".join(lines)


def main():
    nes = NES(ROM)
    boot(nes)
    nes.controller = NES_INPUT_RIGHT

    prev = None
    shown = 0
    for i in range(1500):
        frame = np.asarray(nes.step(), dtype=np.uint8)
        y = R.px(R.read_obj_world_y(nes, 0))
        x = R.px(R.read_obj_world_x(nes, 0))
        if y > 200 and prev is not None and shown == 0:
            print(f"=== 坠落前一帧 (f{i})  player screen x={x:.0f} y={y:.0f} ===")
            print("    x 从 player-16 到 player+96, y 从 160 到 208")
            print("    . =植被/天空   G =地面   # =黑色(坑)   o =亮色")
            print()
            print(amap(prev, int(x) - 16, 160, 112, 48))
            shown = 1
            break
        prev = frame
    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
