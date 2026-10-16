"""Why does the player stop after ~16px when holding RIGHT?
Log position + screenshot every 60 frames.
"""

import os

import numpy as np
from PIL import Image

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"
OUT = r"D:\source_code\adventure-island-ai\outputs\probe"
os.makedirs(OUT, exist_ok=True)


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def px_x(nes, slot=0):
    raw = ((nes[0x0594 + 0x10 * slot] << 16) |
           (nes[0x0584 + 0x10 * slot] << 8) |
           nes[0x0574 + 0x10 * slot])
    return raw / 256.0          # 16.8 fixed point -> pixels


def px_y(nes, slot=0):
    return ((nes[0x05C4 + 0x10 * slot] << 8) |
            nes[0x05B4 + 0x10 * slot]) / 256.0


nes = NES(ROM)
boot(nes)
print(f"boot: x={px_x(nes):.1f}px y={px_y(nes):.1f}px lives={nes[0x3F]}")

print("\n--- hold RIGHT 720f ---")
nes.controller = NES_INPUT_RIGHT
for i in range(720):
    nes.step()
    if i % 60 == 0:
        arr = np.asarray(nes.step(), dtype=np.uint8)
        Image.fromarray(arr, mode="RGB").save(os.path.join(OUT, f"stuck_{i:04d}.png"))
        spd = nes[0x05A4]
        spd = spd - 256 if spd > 127 else spd
        print(f"  f{i:4d} x={px_x(nes):7.1f} y={px_y(nes):6.1f} spd={spd:4d} "
              f"grav={nes[0x05D4]:3d} lives={nes[0x3F]} sta={nes[0x76]:2d}")

print("\n--- now try RIGHT + jump every 40f for 400f ---")
for i in range(400):
    nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if i % 40 < 12 else 0)
    nes.step()
    if i % 100 == 0:
        print(f"  f{i:4d} x={px_x(nes):7.1f} y={px_y(nes):6.1f} spd={nes[0x05A4]:4d}")
nes.controller = 0
arr = np.asarray(nes.step(), dtype=np.uint8)
Image.fromarray(arr, mode="RGB").save(os.path.join(OUT, "stuck_jumping.png"))
print(f"  final x={px_x(nes):.1f}")
del nes
