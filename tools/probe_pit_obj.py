"""Is id 65 the pit marker?  Compare its spawn positions with fall deaths.

Run scripted play (right + periodic jumps, so we survive a while), and
for every id-65 spawn record its WORLD position.  Do the same for every
death.  If deaths line up with id-65 positions, it marks the pits.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot


def worker(seed):
    rng = np.random.default_rng(seed)
    nes = NES(ROM)
    boot(nes)
    cam = R.CameraTracker(nes)
    spawns = []            # world x of id-65 spawns
    deaths = []            # world x at death
    prev = [0] * 16
    p_lives = nes[R.LIVES]

    for f in range(12000):
        ctrl = NES_INPUT_RIGHT
        if (f % 30) < 12:
            ctrl |= NES_INPUT_A          # hop constantly to survive pits
        nes.controller = ctrl
        nes.step()
        base = cam.progress(nes)
        px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        for s in range(16):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid == 65 and prev[s] != 65:
                spawns.append(base + (R.px(R.read_obj_world_x(nes, s)) - px))
            prev[s] = oid
        if nes[R.LIVES] < p_lives:
            deaths.append(round(base, 1))
            p_lives = nes[R.LIVES]
        if nes[R.LIVES] == 0:
            break
    nes.controller = 0
    del nes
    return spawns, deaths


def main():
    all_s, all_d = [], []
    with mp.Pool(6) as pool:
        for s, d in pool.imap_unordered(worker, range(6)):
            all_s += s
            all_d += d
    print(f"id65 生成 {len(all_s)} 次, 死亡 {len(all_d)} 次")
    print()
    print("id65 出生时的世界进度（前 25 个）:")
    print("  ", [round(v) for v in sorted(all_s)[:25]])
    print()
    print("死亡时的世界进度:")
    print("  ", sorted(set(all_d)))
    print()
    if all_s and all_d:
        # for each death, distance to nearest id65 spawn
        ds = [min(abs(d - s) for s in all_s) for d in all_d]
        print("每个死亡点到最近 id65 的距离:", [round(v, 1) for v in sorted(ds)])
        print(f"中位距离: {np.median(ds):.1f}px   (若普遍 < 64px 即强关联)")


if __name__ == "__main__":
    main()
