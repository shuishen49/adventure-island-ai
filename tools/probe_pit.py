"""Dump pixel columns just before the player falls, to learn what a pit looks like."""
import sys

import numpy as np

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot


def ground_y(frame, x):
    """Topmost y of the ground: scan up from the bottom, first brown pixel."""
    for y in range(239, 60, -1):
        r, g, b = (int(v) for v in frame[y, x])
        if r > 60 and r > g and g > b:          # brown-ish (ground tiles)
            continue
        else:
            return y + 1
    return 240


def main():
    nes = NES(ROM)
    boot(nes)
    nes.controller = NES_INPUT_RIGHT

    prev = None
    for i in range(1500):
        frame = np.asarray(nes.step(), dtype=np.uint8)
        y = R.px(R.read_obj_world_y(nes, 0))
        x = R.px(R.read_obj_world_x(nes, 0))
        if y > 200 and prev is not None:          # falling
            print(f"坠落开始于帧 {i}, player screen x={x:.0f} y={y:.0f}")
            print("  坠落前一帧, 玩家右前方各列的地面高度 (x : ground_y) :")
            for off in range(-8, 97, 8):
                cx = int(np.clip(x + off, 0, 255))
                print(f"    x={cx:3d} (dx{off:+3d}) -> ground_y={ground_y(prev, cx)}")
            print()
            print("  坠落前一帧 column x=player+40 的颜色 (y=150..239, 每6行):")
            cx = int(np.clip(x + 40, 0, 255))
            for yy in range(150, 240, 6):
                print(f"    y={yy:3d} {tuple(int(v) for v in prev[yy, cx])}")
            break
        prev = frame
    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
