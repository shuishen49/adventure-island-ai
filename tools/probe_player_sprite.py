"""Identify the player's OAM sprites and derive the camera offset.

Once we know the player's screen X, camera = worldX - screenX, which lets
us project every RAM object (enemies included) onto the screen.
"""

import numpy as np

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START

ROM = r"D:\source_code\cynes\冒险岛1.nes"


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=150)


def oam(nes):
    nes.step()          # advance one frame, then read the sprite table
    return np.fromiter((nes[a] for a in range(0x200, 0x300)),
                       dtype=np.uint8, count=256).astype(int)


def world_x(nes, slot=0):
    return ((nes[0x0594 + 0x10 * slot] << 16) |
            (nes[0x0584 + 0x10 * slot] << 8) |
            nes[0x0574 + 0x10 * slot]) / 16.0


nes = NES(ROM)
boot(nes)

# --- jump experiment: player sprites are the ones whose Y swings a lot ---
nes.controller = NES_INPUT_A
jump = [oam(nes) for _ in range(6)]
nes.controller = 0
jump += [oam(nes) for _ in range(70)]
J = np.stack(jump)

amp = np.array([J[:, 4 * s].max() - J[:, 4 * s].min() for s in range(64)])
x_mean = J[:, 3::4].mean(axis=0)

print("sprites with Y amplitude > 25 during a jump:")
for s in range(64):
    if amp[s] > 25:
        ys = J[:, 4 * s]
        xs = J[:, 4 * s + 3]
        print(f"  slot {s:2d}: Y {ys.min():3d}..{ys.max():3d} (amp {amp[s]:3d})  "
              f"X {xs.min():3d}..{xs.max():3d}  tile={J[0, 4 * s + 1]:3d}")

# camera calibration using the largest-amplitude slot
s_best = int(amp.argmax())
screen_x = float(J[-1, 4 * s_best + 3])
wx = world_x(nes)
print(f"\nbest player sprite slot = {s_best}")
print(f"  worldX={wx:.1f}  screenX={screen_x:.1f}  "
      f"=> camera = worldX - screenX = {wx - screen_x:.1f}")

# check stability: walk right, camera should track worldX
print("\nwalking right, checking camera stability:")
for i in range(6):
    nes.controller = NES_INPUT_RIGHT
    nes.step(frames=90)
    nes.controller = 0
    nes.step(frames=30)
    o = oam(nes)
    sx = o[4 * s_best + 3]
    wx = world_x(nes)
    print(f"  worldX={wx:8.1f}  playerScreenX={sx:4d}  camera={wx - sx:8.1f}")

del nes
