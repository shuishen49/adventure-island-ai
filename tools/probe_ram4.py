"""Probe v4 - OAM sprite table analysis + scroll candidate trajectories.

The $0700-$07FF region is the OAM shadow table: 64 sprites x 4 bytes
(Y, tile, attr, X).  We identify which slots are the player and which
are enemies by watching how each slot moves during:
   - idle      : enemies animate in place
   - run right : world scrolls left relative to screen -> enemy screen X
                 decreases, player X stays put
   - jump      : player slot Y traces a parabola, others do not

Also dumps the raw trajectories of the scroll-candidate addresses so we
can see which one actually accumulates world distance.
"""

import os

import numpy as np
from PIL import Image

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"
OUT = r"D:\source_code\adventure-island-ai\outputs\probe"
os.makedirs(OUT, exist_ok=True)

SCROLL_CANDS = [0x574, 0x5A4, 0x592, 0x593, 0x3C, 0x13C, 0x13B, 0x131,
                0x581, 0x583, 0x631, 0x633]


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=180)


def scan_oam(nes):
    return np.fromiter((nes[a] for a in range(0x700, 0x800)),
                       dtype=np.uint8, count=0x100)


def scan_cands(nes):
    return {a: nes[a] for a in SCROLL_CANDS}


def shot(nes, name):
    arr = np.asarray(nes.step(), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(os.path.join(OUT, f"{name}.png"))


nes = NES(ROM)
boot(nes)

# ---------------------------------------------------------------
# 1. OAM identification experiment
# ---------------------------------------------------------------
print("=== collecting OAM trajectories ===")
segs = {}

nes.controller = 0x00
segs["idle"] = [scan_oam(nes) for _ in range(120)]
nes.controller = 0

nes.controller = NES_INPUT_RIGHT
segs["run"] = [scan_oam(nes) for _ in range(240)]
nes.controller = 0

nes.controller = NES_INPUT_A
segs["jumpA"] = [scan_oam(nes) for _ in range(10)]
nes.controller = 0
segs["jumpB"] = [scan_oam(nes) for _ in range(70)]

idle = np.stack(segs["idle"])
run = np.stack(segs["run"])
jump = np.stack(segs["jumpA"] + segs["jumpB"])

print(f"\n{'slot':>4} | {'idle_x':>16} | {'run_x (start->end)':>22} | "
      f"{'jump_y (min..max)':>20} | verdict")
print("-" * 90)
for s in range(64):
    y_idle = idle[:, 4 * s]
    x_run = run[:, 4 * s + 3]
    y_jump = jump[:, 4 * s]
    # skip sprites that are always off-screen (OAM y=240 hides them)
    if (y_jump == 240).all() and (x_run < 8).all():
        continue
    idle_var = int(y_idle.std() + idle[:, 4 * s + 3].std())
    run_dx = int(x_run[-1].astype(int) - x_run[0].astype(int))
    jump_yr = (int(y_jump.min()), int(y_jump.max()))
    jump_amp = jump_yr[1] - jump_yr[0]

    verdict = ""
    if jump_amp > 30 and idle_var < 40:
        verdict = "<== PLAYER CANDIDATE"
    elif run_dx < -30:
        verdict = "enemy (moves left w/ scroll)"

    print(f"{s:4d} | var={idle_var:4d}        | {int(x_run[0]):4d} -> {int(x_run[-1]):4d} "
          f"(d={run_dx:+4d})  | {jump_yr[0]:3d}..{jump_yr[1]:3d} amp={jump_amp:3d} | {verdict}")

# ---------------------------------------------------------------
# 2. scroll candidate raw trajectories over a long right run
# ---------------------------------------------------------------
print("\n=== scroll candidates over 900f of holding RIGHT ===")
nes.controller = NES_INPUT_RIGHT
traj = {a: [] for a in SCROLL_CANDS}
for i in range(900):
    nes.step()
    if i % 3 == 0:  # sample every 3 frames, cheaper + less noise
        for a in SCROLL_CANDS:
            traj[a].append(nes[a])
nes.controller = 0
shot(nes, "after_900f_right")

print(f"{'addr':>6} {'first':>6} {'last':>6} {'net':>6} {'chg%':>5}")
for a in SCROLL_CANDS:
    t = np.array(traj[a])
    chg = (np.diff(t) != 0).mean() * 100
    print(f"{a:#06x} {t[0]:6d} {t[-1]:6d} {t[-1] - t[0]:+6d} {chg:5.0f}")

del nes
