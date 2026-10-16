"""Is id 66 your own projectile?  Track its spawn instant.

For every object slot, log the frame a slot turns into a given id, where
it appeared relative to the player, which way it moved, and whether the
throw button (B) had been pressed just before.  Run a strict control pass
with B never pressed alongside a pass that throws often.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_A, NES_INPUT_B, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot

WATCH = (66, 14, 9, 28, 16, 17)


def run(throw: bool, frames: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    nes = NES(ROM)
    boot(nes)
    prev_ids = [0] * 16
    spawns = {i: [] for i in WATCH}
    press_hist = [0] * 12           # frames since B was last pressed
    total_frames = {i: 0 for i in WATCH}

    for f in range(frames):
        ctrl = NES_INPUT_RIGHT
        if (f % 41) < 9:
            ctrl |= NES_INPUT_A
        if throw and (f % 23) < 3:
            ctrl |= NES_INPUT_B
        nes.controller = ctrl
        nes.step()
        b_now = 1 if (ctrl & NES_INPUT_B) else 0
        press_hist = [0] + press_hist[:-1]
        if b_now:
            press_hist[0] = 1

        px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        for s in range(16):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid in WATCH:
                total_frames[oid] += 1
            if oid in WATCH and prev_ids[s] != oid:
                x = R.px(R.read_obj_world_x(nes, s))
                y = R.px(R.read_obj_world_y(nes, s))
                spd = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
                spd = spd - 256 if spd > 127 else spd
                spawns[oid].append((f, x - px, y - py, spd,
                                    1 if any(press_hist[:6]) else 0))
            prev_ids[s] = oid

    nes.controller = 0
    del nes
    return spawns, total_frames, frames


def main():
    for throw in (False, True):
        spawns, tot, fr = run(throw)
        tag = "按投掷键(B)" if throw else "从不按 B (对照)"
        print(f"\n===== {tag} — {fr} 帧 =====")
        print("id   出现帧数   生成次数  生成时 dx(相对玩家)  生成时 dy  "
              "初速  投掷后6帧内生成%")
        for oid in WATCH:
            sp = spawns[oid]
            if not sp:
                print(f"{oid:3d} {tot[oid]:9d} {0:9d}   — 从未生成")
                continue
            dx = np.mean([a[1] for a in sp])
            dy = np.mean([a[2] for a in sp])
            v0 = np.mean([a[3] for a in sp])
            pb = 100 * np.mean([a[4] for a in sp])
            print(f"{oid:3d} {tot[oid]:9d} {len(sp):9d}   dx={dx:8.1f}  "
                  f"dy={dy:7.1f}  v0={v0:6.1f}  {pb:6.1f}%")
        print("  前 6 次生成明细 (帧, dx, dy, 初速, 刚按过B):")
        for oid in WATCH:
            for a in spawns[oid][:6]:
                print(f"    id{oid}: frame={a[0]:5d} dx={a[1]:+7.1f} "
                      f"dy={a[2]:+6.1f} v0={a[3]:+5d} B={a[4]}")


if __name__ == "__main__":
    main()
