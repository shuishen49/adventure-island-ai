"""Final verification of the reverse-engineered RAM map.

Object structure (stride 0x10, slot 0 = player):
  +0x10 base  field
  $0564       obj_id
  $0574       pos_X frac
  $0584       pos_X lo
  $0594       pos_X hi
  $05A4       spd_X
  $05B4       pos_Y frac
  $05C4       pos_Y lo
  $05D4       gravity
Statics: $3F lives, $76 stamina, $6D weapon, $6E skateboard,
         $37 area, $38 round, $72 death, $69A score, $0006 scrollX
"""

import os

import numpy as np
from PIL import Image

from cynes import NES, NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT
from cynes import NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"
OUT = r"D:\source_code\adventure-island-ai\outputs\probe"

OBJ_FIELDS = {
    "id": 0x0564, "xfr": 0x0574, "xlo": 0x0584, "xhi": 0x0594,
    "sx": 0x05A4, "yfr": 0x05B4, "ylo": 0x05C4, "grav": 0x05D4,
}
N_SLOTS = 8


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=180)


def world_x(nes, slot=0):
    hi = nes[OBJ_FIELDS["xhi"] + 0x10 * slot]
    lo = nes[OBJ_FIELDS["xlo"] + 0x10 * slot]
    fr = nes[OBJ_FIELDS["xfr"] + 0x10 * slot]
    return (hi << 16) | (lo << 8) | fr


def world_y(nes, slot=0):
    lo = nes[OBJ_FIELDS["ylo"] + 0x10 * slot]
    fr = nes[OBJ_FIELDS["yfr"] + 0x10 * slot]
    return (lo << 8) | fr


def snap(nes, name):
    arr = np.asarray(nes.step(), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(os.path.join(OUT, name))


def status(nes, tag):
    print(f"[{tag:>12}] worldX={world_x(nes) >> 4:6d} scrX={nes[6]:3d} "
          f"Y={world_y(nes) >> 4:5d} spdX={nes[OBJ_FIELDS['sx']]:4d} "
          f"grav={nes[OBJ_FIELDS['grav']]:3d} lives={nes[0x3F]} "
          f"sta={nes[0x76]:2d} score={nes[0x69A]:3d} death={nes[0x72]}")


nes = NES(ROM)
boot(nes)
print("--- after boot ---")
status(nes, "boot")

print("\n--- hold RIGHT 180f ---")
nes.controller = NES_INPUT_RIGHT
for i in range(180):
    nes.step()
    if i in (0, 60, 120, 179):
        status(nes, f"right f{i}")
snap(nes, "v_right.png")

print("\n--- jump (hold A 10f, observe 70f) ---")
nes.controller = NES_INPUT_A
nes.step(frames=10)
nes.controller = 0
for i in range(0, 70):
    nes.step()
    if i % 10 == 0:
        status(nes, f"jump f{i}")

print("\n--- release, stand 60f ---")
nes.step(frames=60)
status(nes, "stand")

print("\n--- hold LEFT 120f (walking back should DECREASE worldX) ---")
nes.controller = NES_INPUT_LEFT
for i in range(120):
    nes.step()
    if i in (0, 60, 119):
        status(nes, f"left f{i}")

print("\n--- object slots (id / worldX>>4 / worldY>>4 / spdX) ---")
for s in range(N_SLOTS):
    base = 0x0564 + 0x10 * s
    oid = nes[base]
    if oid == 0:
        continue
    print(f"  slot {s}: id={oid:3d} X={world_x(nes, s) >> 4:6d} "
          f"Y={world_y(nes, s) >> 4:5d} sx={nes[0x05A4 + 0x10 * s]:4d}")

snap(nes, "v_final.png")
del nes
