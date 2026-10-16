"""Inspect score memory layout."""

from cynes import NES, NES_INPUT_RIGHT, NES_INPUT_START

nes = NES(r"D:\source_code\cynes\冒险岛1.nes")
nes.step(frames=300)
nes.controller = NES_INPUT_START
nes.step(frames=60)
nes.controller = 0
nes.step(frames=60)

print("boot score region 0x698-0x6AC:")
for a in range(0x698, 0x6AD):
    print(f"  {a:#06x} = {nes[a]:3d} (0x{nes[a]:02x})")

print("\nafter running right 900f:")
nes.controller = NES_INPUT_RIGHT
nes.step(frames=900)
nes.controller = 0
for a in range(0x698, 0x6AD):
    print(f"  {a:#06x} = {nes[a]:3d} (0x{nes[a]:02x})")
del nes
