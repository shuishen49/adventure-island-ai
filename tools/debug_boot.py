"""Debug: figure out how to leave the title screen and enter actual gameplay."""

import os

import numpy as np
from PIL import Image

from cynes import NES, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"
OUT = r"D:\source_code\adventure-island-ai\outputs\probe"
os.makedirs(OUT, exist_ok=True)

WATCH = [0x3F, 0x76, 0x6D, 0x6E, 0x75, 0x37, 0x38, 0x39, 0x40, 0x49]


def shot(nes, tag):
    arr = np.asarray(nes.step(), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").resize((512, 480), Image.NEAREST).save(
        os.path.join(OUT, f"{tag}.png"))
    return arr


def row(nes, i):
    vals = " ".join(f"{nes[a]:3d}" for a in WATCH)
    print(f"  f{i:4d} | {vals}")


nes = NES(ROM)

print("=== phase 1: settle on title screen ===")
for _ in range(300):
    nes.step()
shot(nes, "01_title")
row(nes, 0)

print("\n=== phase 2: hold START for 60 frames ===")
nes.controller = NES_INPUT_START
for i in range(60):
    nes.step()
    if i % 20 == 0:
        row(nes, i)
nes.controller = 0
shot(nes, "02_after_start")

print("\n=== phase 3: release, run 600 frames ===")
for i in range(600):
    nes.step()
    if i % 100 == 0:
        row(nes, i)
shot(nes, "03_600f_later")

print("\n=== phase 4: mash START repeatedly ===")
for burst in range(6):
    nes.controller = NES_INPUT_START
    for _ in range(6):
        nes.step()
    nes.controller = 0
    for _ in range(30):
        nes.step()
    print(f"  burst {burst}: lives={nes[0x3F]} time={nes[0x76]} "
          f"round={nes[0x38]} ckpt={nes[0x39]}")
shot(nes, "04_after_mash")

print("\nwatched:", " ".join(f"{a:#06x}" for a in WATCH))
print("labels : lives time weap skate inv area rnd ckpt demo bonus")
del nes
