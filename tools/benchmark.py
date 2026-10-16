"""Benchmark cynes headless speed and RAM read speed."""

import time

import numpy as np

from cynes import NES

ROM = r"D:\source_code\cynes\冒险岛1.nes"

nes = NES(ROM)

# --- 1. raw frame stepping speed ---
N = 2000
t0 = time.perf_counter()
for _ in range(N):
    nes.step()
t1 = time.perf_counter()
fps = N / (t1 - t0)
print(f"[step]      {N} frames in {t1 - t0:.2f}s  ->  {fps:,.0f} FPS  ({fps / 60.0988:.1f}x realtime)")

# --- 2. RAM read speed: full 2KB scan ---
t0 = time.perf_counter()
for _ in range(200):
    ram = np.fromiter((nes[a] for a in range(0x800)), dtype=np.uint8, count=0x800)
t1 = time.perf_counter()
per_scan = (t1 - t0) / 200
print(f"[ram scan]  2048 B in {per_scan * 1000:.2f} ms -> overhead "
      f"{per_scan / (1 / fps) * 100:.0f}% of a frame")

# --- 3. combined realistic throughput ---
t0 = time.perf_counter()
for _ in range(1000):
    nes.step()
    ram = np.fromiter((nes[a] for a in range(0x800)), dtype=np.uint8, count=0x800)
t1 = time.perf_counter()
print(f"[step+scan] {1000 / (t1 - t0):,.0f} FPS combined")

print(f"\n=> ~{1000 / (t1 - t0) * 60 * 60 / 60.0988:,.0f} game-seconds per wall-minute (per core)")
del nes
