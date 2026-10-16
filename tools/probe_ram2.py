"""Probe v2 - controlled experiments after a CORRECT boot sequence.

Boot fix: hold START for 60 frames (8 was too short), then settle.

Experiments:
  A  idle 240f        - baseline noise (timers, RNG, hud flicker)
  B  hold RIGHT 480f  - scroll / global X should grow monotonically
  C  jump spam while idle 240f - vertical position / vy / jump state
  D  idle 120f then RIGHT 120f - differential view
"""

import numpy as np

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"
RAM_END = 0x800

KNOWN = {0x3F: "lives", 0x76: "time", 0x6D: "weapon", 0x6E: "skateboard",
         0x75: "invinc", 0x37: "area", 0x38: "round", 0x39: "ckpt"}


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=180)


def scan(nes):
    return np.fromiter((nes[a] for a in range(RAM_END)), dtype=np.uint8, count=RAM_END)


def run(nes, ctrl, frames):
    nes.controller = ctrl
    out = []
    for _ in range(frames):
        nes.step()
        out.append(scan(nes))
    nes.controller = 0
    return np.stack(out).astype(np.int16)


def describe(tag, snaps, ref=None, top=25, sort="mono"):
    """Print addresses that move.  If ref given, only show addresses that
    are significantly busier than ref (differential view)."""
    chg = (np.diff(snaps, axis=0) != 0).sum(axis=0)
    d = np.diff(snaps, axis=0)
    mono = np.abs(d.sum(axis=0)) / (np.abs(d).sum(axis=0) + 1e-9)
    lo, hi = snaps.min(axis=0), snaps.max(axis=0)
    print(f"\n{'=' * 84}\n{tag}\n{'=' * 84}")
    print(f"{'addr':>6} {'chg':>4} {'mono':>5} {'lo':>4} {'hi':>4} {'last':>5}  note")
    idx = np.argsort(-chg if sort == "chg" else -mono)
    n = 0
    for a in idx:
        if chg[a] < 6:
            continue
        if ref is not None and chg[a] <= ref[a] + 5:
            continue
        note = KNOWN.get(int(a), "")
        print(f"{a:#06x} {chg[a]:4d} {mono[a]:5.2f} {lo[a]:4d} {hi[a]:4d} "
              f"{snaps[-1, a]:5d}  {note}")
        n += 1
        if n >= top:
            break


nes = NES(ROM)
boot(nes)
print("booted:", {KNOWN[a]: nes[a] for a in KNOWN})

A = run(nes, 0x00, 240)
B = run(nes, NES_INPUT_RIGHT, 480)
C = run(nes, NES_INPUT_A, 240)  # jump spam while standing

# address-wise change counts of the idle baseline, reused for filtering
chgA = (np.diff(A, axis=0) != 0).sum(axis=0)

describe("A  idle 240f", A, top=15)
describe("B  RIGHT 480f (filtered: busier than idle)", B, ref=chgA, top=25, sort="mono")
describe("C  jump spam 240f (filtered)", C, ref=chgA, top=25, sort="chg")

# --- monotonic growers in B -> scroll / world X candidates ---
d = np.diff(B, axis=0)
print(f"\n{'=' * 84}\nmonotonic INCREASE during RIGHT run (scroll / world-x candidates)\n{'=' * 84}")
print(f"{'addr':>6} {'total':>6} {'mono':>5} {'lo':>4} {'hi':>4} {'last':>6}")
for a in range(RAM_END):
    col = B[:, a]
    dd = d[:, a]
    if dd.sum() < 40 or chg[a] < 30:
        continue
    m = dd.sum() / (np.abs(dd).sum() + 1e-9)
    if m < 0.85:
        continue
    print(f"{a:#06x} {dd.sum():6d} {m:5.2f} {col.min():4d} {col.max():4d} {col[-1]:6d}")

# --- jump-only movers ---
print(f"\n{'=' * 84}\njump-spam movers (Y position / vy candidates)\n{'=' * 84}")
print(f"{'addr':>6} {'A_chg':>5} {'C_chg':>5} {'lo':>4} {'hi':>4} {'last':>5}")
chgC = (np.diff(C, axis=0) != 0).sum(axis=0)
for a in np.argsort(-chgC):
    if chgC[a] < 20 or chgA[a] > chgC[a] - 10:
        continue
    if 0x0600 <= a < 0x0800:  # skip oam-ish noise for now
        continue
    print(f"{a:#06x} {chgA[a]:5d} {chgC[a]:5d} {C[:, a].min():4d} "
          f"{C[:, a].max():4d} {C[-1, a]:5d}")

del nes
