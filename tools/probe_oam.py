"""Locate the OAM shadow buffer (sprite table).

If a $xx00-$xxFF page behaves like sprites (Y in 16..240, X in 0..255,
4-byte stride) we can draw exact screen-space boxes for every object.
Also calibrate: does worldX - scroll give the player's screen X?
"""

import numpy as np

from cynes import NES, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def page(nes, base):
    return np.fromiter((nes[a] for a in range(base, base + 256)),
                       dtype=np.uint8, count=256)


nes = NES(ROM)
boot(nes)

print("=== scanning 256-byte pages for OAM-like layout ===")
print("(a sprite page: byte0 = Y in 16..240, byte3 = X in 0..255, stride 4)")
for base in range(0x0000, 0x0800, 0x100):
    p = page(nes, base).astype(int)
    ys = p[0::4]
    xs = p[3::4]
    y_ok = ((ys >= 8) & (ys <= 240)).mean()
    x_ok = ((xs <= 255)).mean()
    if y_ok > 0.6 and (ys != 240).sum() > 4 and (ys != 0).sum() > 4:
        print(f"  {base:#06x}: y_in_range={y_ok:.2f} non_hidden={int((ys != 240).sum()):3d} "
              f"ys={list(ys[:12])} xs={list(xs[:12])}")

print("\n=== calibrate scroll: player screen X vs worldX and $0006 ===")
for step_i in range(4):
    nes.controller = NES_INPUT_RIGHT
    nes.step(frames=120)
    nes.controller = 0
    nes.step(frames=30)
    hi = nes[0x0594]; lo = nes[0x0584]; fr = nes[0x0574]
    wx = ((hi << 16) | (lo << 8) | fr) / 16.0
    print(f"  worldX={wx:8.1f}  $0006={nes[6]:3d}  $0007={nes[7]:3d}  "
          f"$0008={nes[8]:3d}  $0009={nes[9]:3d}  $002A={nes[0x2A]:3d} "
          f"$002B={nes[0x2B]:3d}")

del nes
