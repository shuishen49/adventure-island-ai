"""Look at the pixels right in front of the player to see what blocks it."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = r"D:\source_code\cynes\冒险岛1.nes"


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


nes = NES(ROM)
boot(nes)
nes.controller = NES_INPUT_RIGHT
nes.step(frames=200)          # run into the wall
nes.controller = 0
nes.step(frames=20)
frame = np.asarray(nes.step(), dtype=np.uint8)

x = R.px(R.read_obj_world_x(nes, 0))
y = R.px(R.read_obj_world_y(nes, 0))
print(f"player at x={x:.1f} y={y:.1f}")

print("\ncolumn colours at the player's feet/body rows")
print("(x from 80 to 150 step 5, rows 110..170 step 10)")
print(f"{'x':>5}", "".join(f"{'y'+str(yy):>16}" for yy in range(110, 171, 10)))
for xx in range(80, 151, 5):
    cells = []
    for yy in range(110, 171, 10):
        r, g, b = frame[yy, xx]
        cells.append(f"{r:4d},{g:3d},{b:3d}")
    print(f"{xx:5d}", " ".join(f"{c:>16}" for c in cells))

# how much of the row at the player's feet is "sky" (uniform) vs solid
print("\nrow 150 (feet level) - is anything solid to the right?")
row = frame[150, 80:160]
uniq, counts = np.unique(row.reshape(-1, 3), axis=0, return_counts=True)
top = uniq[np.argsort(-counts)][:5]
print("  dominant colours:", [tuple(int(v) for v in c) for c in top])

print("\nvertical slice at x=100 (just right of the wall):")
for yy in range(100, 175, 5):
    r, g, b = frame[yy, 100]
    print(f"  y={yy:3d}  rgb=({r:3d},{g:3d},{b:3d})")

del nes
