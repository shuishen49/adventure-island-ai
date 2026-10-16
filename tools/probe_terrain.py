"""Inspect the pixel column profile ahead of the player, especially at pits."""
import sys

import numpy as np

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot


def sky_color(frame):
    return tuple(int(v) for v in frame[3, 3])


def col_profile(frame, x, sky, tol=40):
    """First non-sky y scanning down from y=40; 240 if none."""
    for y in range(40, 240):
        p = frame[y, x]
        if abs(int(p[0]) - sky[0]) + abs(int(p[1]) - sky[1]) + \
           abs(int(p[2]) - sky[2]) > tol:
            return y
    return 240


def main():
    nes = NES(ROM)
    boot(nes)
    cam = R.CameraTracker(nes)

    print("phase 1: 正常地面剖面")
    nes.controller = NES_INPUT_RIGHT
    nes.step(frames=200)
    frame = np.asarray(nes.step(), dtype=np.uint8)
    sky = sky_color(frame)
    px = R.px(R.read_obj_world_x(nes, 0))
    print(f"  sky={sky}  player screen x={px:.0f}  progress={cam.progress(nes):.0f}")
    for off in (-16, 0, 16, 32, 48, 64, 80, 96):
        x = int(np.clip(px + off, 0, 255))
        print(f"   x={x:3d} (dx{off:+4d})  ground_y={col_profile(frame, x, sky)}")

    print()
    print("phase 2: 一路跑到坑边，每 40 帧打印一次前方地面剖面")
    for block in range(14):
        for i in range(40):
            nes.controller = NES_INPUT_RIGHT
            frame = np.asarray(nes.step(), dtype=np.uint8)
        px = R.px(R.read_obj_world_x(nes, 0))
        prof = [col_profile(frame, int(np.clip(px + o, 0, 255)), sky)
                for o in (8, 24, 40, 56, 72)]
        y = R.px(R.read_obj_world_y(nes, 0))
        print(f"  f{(block+1)*40+200:4d} prog={cam.progress(nes):6.0f} "
              f"playerY={y:6.1f}  ground@[+8,+24,+40,+56,+72]={prof}")

    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
