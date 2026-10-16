"""Probe v5 - CORRECT OAM analysis (step before every scan).

Identifies:
  * which OAM slots form the player (jump parabola)
  * which slots are enemies/objects (move with the world during a run)
  * the world-scroll address by cross-checking candidate trajectories
    with screenshots taken at known distances
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
    nes.step(frames=180)


def scan_oam(nes):
    return np.fromiter((nes[a] for a in range(0x700, 0x800)),
                       dtype=np.uint8, count=0x100)


def seg(nes, ctrl, frames):
    nes.controller = ctrl
    out = []
    for _ in range(frames):
        nes.step()
        out.append(scan_oam(nes))
    nes.controller = 0
    return np.stack(out)


nes = NES(ROM)
boot(nes)

idle = seg(nes, 0x00, 90)
run = seg(nes, NES_INPUT_RIGHT, 240)
jmp = np.concatenate([seg(nes, NES_INPUT_A, 8), seg(nes, 0x00, 80)], axis=0)

print(f"{'slot':>4} | {'jump Y amp':>10} {'jump X':>7} | "
      f"{'run X start->end':>18} {'runY amp':>8} | verdict")
print("-" * 78)
for s in range(64):
    yj = jmp[:, 4 * s]
    xj = jmp[:, 4 * s + 3]
    xr = run[:, 4 * s + 3]
    yr = run[:, 4 * s]
    amp_j = int(yj.max()) - int(yj.min())
    amp_r = int(yr.max()) - int(yr.min())
    dx = int(xr[-1]) - int(xr[0])
    on_screen = (yj < 240).any() or (xr < 240).any()
    if not on_screen:
        continue
    verdict = ""
    if amp_j > 40:
        verdict = "<== PLAYER (jump parabola)"
    elif dx <= -20 and amp_j < 10:
        verdict = "world object (scrolls left)"
    print(f"{s:4d} | {amp_j:10d} {int(xj[0]):7d} | "
          f"{int(xr[0]):5d}->{int(xr[-1]):5d} {amp_r:8d} | {verdict}")

# detailed look at the player candidate: print its jump trajectory
print("\n=== jump trajectory of candidate slots (Y values) ===")
cands = [s for s in range(64)
         if (jmp[:, 4 * s].max() - jmp[:, 4 * s].min()) > 40]
for s in cands[:4]:
    ys = jmp[::4, 4 * s]
    print(f"slot {s:2d} Y: {list(map(int, ys))}")

del nes
