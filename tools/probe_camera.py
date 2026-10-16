"""Find the camera/scroll address by correlating every RAM byte against
the player's world X over a long run.  The camera must track worldX
almost perfectly (|r| > 0.95).
"""

import numpy as np

from cynes import NES, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"
RAM_END = 0x800


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def world_x(nes):
    return ((nes[0x0594] << 16) | (nes[0x0584] << 8) | nes[0x0574]) / 16.0


nes = NES(ROM)
boot(nes)

# keep the player moving so the camera actually scrolls
nes.controller = NES_INPUT_RIGHT
wx, ram = [], []
for i in range(900):
    nes.step()
    wx.append(world_x(nes))
    ram.append(np.fromiter((nes[a] for a in range(RAM_END)),
                           dtype=np.uint8, count=RAM_END))
nes.controller = 0

wx = np.array(wx)
ram = np.stack(ram).astype(np.float64)
print(f"worldX range: {wx.min():.0f} .. {wx.max():.0f} (moved {wx.max() - wx.min():.0f}px)")

std = ram.std(axis=0)
print("\naddresses most correlated with worldX (camera / scroll candidates):")
print(f"{'addr':>6} {'r':>7} {'lo':>4} {'hi':>4} {'first':>6} {'last':>6}")
rows = []
for a in range(RAM_END):
    if std[a] < 1e-6 or wx.std() < 1e-6:
        continue
    # a 16-bit counter wraps; treat wrap by unwrapping before correlating
    col = ram[:, a]
    if col.max() > 200 and col.min() < 55:          # likely wrapped
        col = np.unwrap(col * (2 * np.pi / 256)) / (2 * np.pi / 256)
    r = np.corrcoef(col, wx)[0, 1]
    if abs(r) > 0.9:
        rows.append((a, r, int(ram[:, a].min()), int(ram[:, a].max()),
                     int(ram[0, a]), int(ram[-1, a])))
for a, r, lo, hi, f0, f1 in sorted(rows, key=lambda x: -abs(x[1])):
    print(f"{a:#06x} {r:+7.3f} {lo:4d} {hi:4d} {f0:6d} {f1:6d}")

del nes
