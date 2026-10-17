"""Probe: can we warp directly to Round 1-2 (the second stage)?

Three candidate tricks, each on a fresh NES instance:

  A. set $0038 (ROUND) = 2 while the game initialises stage 1-1
     (right after the title-screen START press)
  B. set $007A (STAGE_COMPLETE) = 1 during stage 1-1 gameplay and let the
     game run its own clear -> next-round transition
  C. set $0038 = 2 in the middle of stage 1-1 gameplay (control test:
     should do nothing until the next stage load)

For each: log area/round/lives/checkpoint + player slot over time and dump
screenshots so the result can be eyeballed.

usage: python tools/probe_round2_warp.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = os.path.join(_ROOT, "roms", "Adventure Island (USA).nes")
OUT = os.path.join(_ROOT, "outputs", "probe_round2")
os.makedirs(OUT, exist_ok=True)

AREA, ROUND, CKPT, LIVES = R.AREA, R.ROUND, R.CHECKPOINT, R.LIVES
CLEAR, DEATH = R.STAGE_COMPLETE, R.DEATH


def shot(nes, tag):
    arr = np.asarray(nes.step(), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").resize((512, 480), Image.NEAREST).save(
        os.path.join(OUT, f"{tag}.png"))
    return arr


def log(nes, i, tag=""):
    pid = nes[R.OBJ.id + R.PLAYER_SLOT]
    x = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    print(f"  f{i:5d} {tag:<10} area={nes[AREA]} round={nes[ROUND]} "
          f"ckpt={nes[CKPT]} lives={nes[LIVES]} clear={nes[CLEAR]} "
          f"death={nes[DEATH]} pid={pid} px=({x:6.1f},{y:6.1f})",
          flush=True)


def boot(nes):
    """Title screen -> START -> stage 1-1 gameplay, like env bootstrap."""
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=60)


# ----------------------------------------------------------------------
print("=== A: force ROUND=2 during stage init (right after START) ===")
nes = NES(ROM)
nes.step(frames=300)
nes.controller = NES_INPUT_START
for i in range(150):
    nes[ROUND] = 2                      # hammer it every frame
    nes.step()
nes.controller = 0
log(nes, 0, "init+150f")
shot(nes, "A_init")
for i in range(120):
    nes.step()
    if i % 40 == 0:
        log(nes, i, "settle")
shot(nes, "A_settle")
del nes

# ----------------------------------------------------------------------
print("\n=== B: force STAGE_COMPLETE=1 during stage 1-1 gameplay ===")
nes = NES(ROM)
boot(nes)
log(nes, 0, "gameplay")
shot(nes, "B_before")
nes[CLEAR] = 1
for i in range(30):
    nes.step()
log(nes, 30, "flag=1")
for i in range(600):
    nes.step()
    if i % 60 == 0:
        log(nes, i, "after")
    if nes[ROUND] != 1:
        print(f"  --> ROUND changed to {nes[ROUND]} at +{i}f")
        break
shot(nes, "B_after")
del nes

# ----------------------------------------------------------------------
print("\n=== C: set ROUND=2 mid-gameplay (control) ===")
nes = NES(ROM)
boot(nes)
for _ in range(120):                    # run right a bit
    nes.controller = 0x01
    nes.step()
nes.controller = 0
log(nes, 0, "midgame")
nes[ROUND] = 2
for i in range(240):
    nes.step()
    if i % 60 == 0:
        log(nes, i, "warp?")
shot(nes, "C_after")
del nes

print("\nscreenshots in", OUT)
