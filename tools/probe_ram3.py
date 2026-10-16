"""Probe v3 - screenshot-correlated experiments.

Fixes the chg bug and, crucially, captures screenshots at experiment
boundaries so we can visually confirm the player actually moved.

Exp 1: single jump (parabola detection -> player Y, vy)
Exp 2: hold RIGHT (screenshot before/after -> confirms motion,
       candidates for scroll / world X)
"""

import os

import numpy as np
from PIL import Image

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"
OUT = r"D:\source_code\adventure-island-ai\outputs\probe"
RAM_END = 0x800
os.makedirs(OUT, exist_ok=True)


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=180)


def scan(nes):
    return np.fromiter((nes[a] for a in range(RAM_END)), dtype=np.uint8, count=RAM_END)


def run(nes, ctrl, frames, snap_at=()):
    nes.controller = ctrl
    out, shots = [], {}
    for i in range(frames):
        if i in snap_at:
            arr = np.asarray(nes.step(), dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(
                os.path.join(OUT, f"exp{len(out) and 0 or 0}_{tag}_{i:04d}.png"))
            out.append(scan(nes))
        else:
            nes.step()
            out.append(scan(nes))
    nes.controller = 0
    return np.stack(out).astype(np.int16)


def parabola_candidates(snaps, baseline=None):
    """Addresses whose value goes up then back down - the jump signature."""
    chg = (np.diff(snaps, axis=0) != 0).sum(axis=0)
    lo, hi = snaps.min(axis=0), snaps.max(axis=0)
    results = []
    for a in range(RAM_END):
        col = snaps[:, a].astype(np.float64)
        if chg[a] < 15:
            continue
        peak = int(np.argmax(col))
        if peak < 5 or peak > len(col) - 5:
            continue
        # rising before peak, falling after -> correlation with a triangle
        tri = np.concatenate([np.arange(peak), np.arange(len(col) - peak)[::-1]])
        c = np.corrcoef(col, tri)[0, 1] if col.std() > 1e-6 else 0
        if c > 0.75:
            results.append((a, int(chg[a]), int(lo[a]), int(hi[a]), float(c)))
    return results


nes = NES(ROM)
boot(nes)

# ---------------- exp 1: single jump ----------------
print("=== EXP1: 60f settle, 8f A press, 90f observe ===")
tag = "jump"
E = run(nes, 0x00, 60)
J = run(nes, NES_INPUT_A, 8)
K = run(nes, 0x00, 90)
snaps = np.concatenate([E, J, K], axis=0)
print("jump parabola candidates (addr, chg, lo, hi, corr):")
for r in parabola_candidates(snaps):
    print(f"  {r[0]:#06x} chg={r[1]:3d} lo={r[2]:3d} hi={r[3]:3d} corr={r[4]:.3f}")

# ---------------- exp 2: hold right ----------------
print("\n=== EXP2: hold RIGHT 480f, screenshots at 0/240/479 ===")
tag = "right"
R = run(nes, NES_INPUT_RIGHT, 480, snap_at=(0, 240, 479))
d = np.diff(R, axis=0)
chgR = (d != 0).sum(axis=0)
chgIdle = (np.diff(E, axis=0) != 0).sum(axis=0)

print("\nmonotonic growers while running RIGHT:")
print(f"{'addr':>6} {'chg':>4} {'mono':>5} {'lo':>4} {'hi':>4} {'last':>5}")
for a in range(RAM_END):
    if chgR[a] < 40 or chgR[a] <= chgIdle[a] + 20:
        continue
    m = d[:, a].sum() / (np.abs(d[:, a]).sum() + 1e-9)
    if m < 0.8:
        continue
    print(f"{a:#06x} {chgR[a]:4d} {m:5.2f} {R[:, a].min():4d} "
          f"{R[:, a].max():4d} {R[-1, a]:5d}")

print("\nmost-changed addresses while running RIGHT (differential vs idle):")
print(f"{'addr':>6} {'idle':>5} {'right':>5} {'lo':>4} {'hi':>4} {'last':>5}")
for a in np.argsort(-chgR)[:25]:
    if chgR[a] <= chgIdle[a] + 10:
        continue
    print(f"{a:#06x} {chgIdle[a]:5d} {chgR[a]:5d} {R[:, a].min():4d} "
          f"{R[:, a].max():4d} {R[-1, a]:5d}")

print("\nknown:", {f"{a:#06x}": nes[a] for a in (0x3F, 0x76, 0x38)})
del nes
