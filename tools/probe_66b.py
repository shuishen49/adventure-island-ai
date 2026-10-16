"""Full trajectory log for suspicious object ids.

Groups each slot's contiguous run of one id into a "life": where it was
born relative to the player, which way it travelled, how fast, how long
it lived, and what the player's weapon counter was at birth.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot

WATCH = (66, 14, 9, 28, 16, 17, 2, 43, 19, 18, 25, 65, 69)


def worker(seed_frames):
    seed, frames = seed_frames
    rng = np.random.default_rng(seed)
    nes = NES(ROM)
    boot(nes)
    lives = defaultdict(list)          # oid -> list of life dicts
    open_run = {}
    ctrl = NES_INPUT_RIGHT
    until = 0

    for f in range(frames):
        if f >= until:
            r = rng.random()
            if r < 0.5:
                ctrl = NES_INPUT_RIGHT
            elif r < 0.65:
                ctrl = NES_INPUT_RIGHT | NES_INPUT_A
            elif r < 0.75:
                ctrl = NES_INPUT_A
            elif r < 0.85:
                ctrl = NES_INPUT_LEFT
            elif r < 0.95:
                ctrl = 0
            else:
                ctrl = NES_INPUT_RIGHT | NES_INPUT_B
            until = f + int(rng.integers(6, 26))
        nes.controller = ctrl
        nes.step()

        px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        wpn = nes[R.WEAPON]
        for s in range(16):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            key = (s, oid)
            if oid == 0:
                if s in open_run:
                    lives[open_run[s]["oid"]].append(open_run.pop(s))
                continue
            if oid not in WATCH:
                if s in open_run:
                    lives[open_run[s]["oid"]].append(open_run.pop(s))
                continue
            x = R.px(R.read_obj_world_x(nes, s))
            y = R.px(R.read_obj_world_y(nes, s))
            spd = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
            spd = spd - 256 if spd > 127 else spd
            if key != open_run.get(s, {}).get("key"):
                if s in open_run:
                    lives[open_run[s]["oid"]].append(open_run.pop(s))
                open_run[s] = {"key": key, "oid": oid, "f0": f,
                               "dx0": x - px, "dy0": y - py, "y0": y,
                               "spd": [], "wpn": wpn, "b": int(bool(ctrl & NES_INPUT_B)),
                               "dxs": []}
            run = open_run[s]
            run["spd"].append(spd)
            run["dxs"].append(x - px)
            run["last"] = f
    for s, run in open_run.items():
        lives[run["oid"]].append(run)
    nes.controller = 0
    del nes
    return {k: v for k, v in lives.items()}


def main():
    jobs = [(100 + i, 22000) for i in range(6)]
    all_lives = defaultdict(list)
    with mp.Pool(processes=6) as pool:
        for res in pool.imap_unordered(worker, jobs):
            for oid, lst in res.items():
                all_lives[oid].extend(lst)

    print("id   出现次数  出生dx    出生dy   出生y   平均速度  寿命帧  "
          "出生时武器  距离变化趋势")
    print("-" * 92)
    for oid in sorted(all_lives):
        L = all_lives[oid]
        if not L:
            continue
        dx0 = np.mean([a["dx0"] for a in L])
        dy0 = np.mean([a["dy0"] for a in L])
        y0 = np.mean([a["y0"] for a in L])
        sp = np.mean([np.mean(a["spd"]) for a in L])
        life = np.mean([a["last"] - a["f0"] for a in L])
        wpn = np.mean([a["wpn"] for a in L])
        trend = []
        for a in L[:40]:
            d = a["dxs"]
            if len(d) >= 20:
                trend.append(d[-1] - d[0])
        tr = np.mean(trend) if trend else 0.0
        print(f"{oid:3d} {len(L):9d} {dx0:8.1f} {dy0:8.1f} {y0:7.1f} "
              f"{sp:9.1f} {life:7.1f} {wpn:9.1f} {tr:+9.1f}")

    print()
    print("id 66 的 8 次完整生命周期（出生距离 → 消失时距离，速度采样）:")
    for a in all_lives.get(66, [])[:8]:
        d = a["dxs"]
        samp = d[::max(1, len(d) // 6)][:6]
        print(f"  born dx={a['dx0']:+7.1f} dy={a['dy0']:+6.1f} y={a['y0']:6.1f} "
              f"wpn={a['wpn']} B={a['b']} 寿命={a['last']-a['f0']:3d}  "
              f"dx轨迹={[round(v) for v in samp]}  "
              f"速度={a['spd'][:4]}")
    print()
    print("id 14 的 6 次完整生命周期（对照组 - 已知敌人）:")
    for a in all_lives.get(14, [])[:6]:
        d = a["dxs"]
        samp = d[::max(1, len(d) // 6)][:6]
        print(f"  born dx={a['dx0']:+7.1f} dy={a['dy0']:+6.1f} y={a['y0']:6.1f} "
              f"wpn={a['wpn']} B={a['b']} 寿命={a['last']-a['f0']:3d}  "
              f"dx轨迹={[round(v) for v in samp]}  速度={a['spd'][:4]}")


if __name__ == "__main__":
    main()
