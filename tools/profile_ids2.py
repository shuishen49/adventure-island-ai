"""Rebuild the object-id catalogue with trustworthy evidence.

Why a rewrite: the earlier run inferred "hurts you" from a stamina drop
after contact, but stamina drains on its own all the time, so anything
spawning on top of the player (your own axe!) scored as an enemy.

This version uses signals that cannot happen spontaneously:
  * lives decreasing   -> something lethal was touched
  * score increasing   -> something was collected
  * stamina INCREASING -> a healing item was touched
and an input ablation: run identical random play WITH and WITHOUT the
throw button; ids that only appear when throwing are your own projectiles.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from cynes import (NES, NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT,
                   NES_INPUT_RIGHT)

import ram_map as R
from env import ROM
from visualize import boot

CONTACT_DX, CONTACT_DY = 18, 26
LOOKBACK_DEATH, LOOKBACK_PICK = 30, 12


def _player(nes):
    x = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    return x, y


def _slot_xy(nes, s):
    x = R.px(R.read_obj_world_x(nes, s))
    y = R.px(R.read_obj_world_y(nes, s))
    return x, y


def worker(args):
    seed, frames, mode = args
    rng = np.random.default_rng(seed)
    nes = NES(ROM)
    boot(nes)

    stats = defaultdict(lambda: np.zeros(7))   # fr,mov,sy,sdx,sadx,sdy,saspd
    seen = defaultdict(int)                    # oid -> frames present
    death_attr = defaultdict(int)
    score_attr = defaultdict(int)
    staup_attr = defaultdict(int)
    wpn_attr = defaultdict(int)

    ctrl = NES_INPUT_RIGHT
    hold, hold_until = ctrl, 0
    recent = []                                # (frame, oid) contacts
    p_lives = nes[R.LIVES]
    p_score = nes[0x06A1]
    p_sta = nes[R.STAMINA]
    p_wpn = nes[R.WEAPON]
    frame = 0

    for f in range(frames):
        if f >= hold_until:
            r = rng.random()
            if r < 0.55:
                ctrl = NES_INPUT_RIGHT
            elif r < 0.70:
                ctrl = NES_INPUT_RIGHT | NES_INPUT_A
            elif r < 0.80:
                ctrl = NES_INPUT_A
            elif r < 0.90:
                ctrl = NES_INPUT_LEFT
            else:
                ctrl = 0
            if mode == "throw" and rng.random() < 0.25:
                ctrl |= NES_INPUT_B
            hold_until = f + int(rng.integers(6, 26))
        nes.controller = ctrl
        nes.step()
        frame += 1

        px, py = _player(nes)
        touched = set()
        for s in range(16):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid == 0:
                continue
            seen[oid] += 1
            x, y = _slot_xy(nes, s)
            spd = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
            spd = spd - 256 if spd > 127 else spd
            st = stats[oid]
            st[0] += 1
            st[1] += 1 if spd != 0 else 0
            st[2] += y
            st[3] += x - px
            st[4] += abs(x - px)
            st[5] += y - py
            st[6] += abs(spd)
            if abs(x - px) < CONTACT_DX and abs(y - py) < CONTACT_DY:
                touched.add(oid)
        if R.PLAYER_SLOT:
            pass
        for oid in touched:
            recent.append((frame, oid))
        recent = [(fr, o) for fr, o in recent if frame - fr <= LOOKBACK_DEATH]

        lv, sc = nes[R.LIVES], nes[0x06A1]
        sta, wpn = nes[R.STAMINA], nes[R.WEAPON]
        if lv < p_lives:
            for fr, o in recent:
                if frame - fr <= LOOKBACK_DEATH:
                    death_attr[o] += 1
            recent = []
        if sc != p_score:
            for fr, o in recent:
                if frame - fr <= LOOKBACK_PICK:
                    score_attr[o] += 1
        if sta > p_sta:
            for fr, o in recent:
                if frame - fr <= LOOKBACK_PICK:
                    staup_attr[o] += 1
        if wpn > p_wpn:
            for fr, o in recent:
                if frame - fr <= LOOKBACK_PICK:
                    wpn_attr[o] += 1
        p_lives, p_score, p_sta, p_wpn = lv, sc, sta, wpn

    nes.controller = 0
    del nes
    return (mode, dict(seen), {k: v.tolist() for k, v in stats.items()},
            dict(death_attr), dict(score_attr), dict(staup_attr),
            dict(wpn_attr), frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = []
    for i in range(args.workers):
        jobs.append((1000 + i, args.frames, "nothrow"))
        jobs.append((2000 + i, args.frames, "throw"))

    seen = {"nothrow": defaultdict(int), "throw": defaultdict(int)}
    stats = defaultdict(lambda: np.zeros(7))
    death, score, staup, wpn = (defaultdict(int) for _ in range(4))
    total = {"nothrow": 0, "throw": 0}

    with mp.Pool(processes=min(args.workers, 16)) as pool:
        for mode, sn, st, da, sa, ua, wa, fr in pool.imap_unordered(worker, jobs):
            for k, v in sn.items():
                seen[mode][k] += v
            for k, v in st.items():
                stats[k] += np.asarray(v)
            for k, v in da.items():
                death[k] += v
            for k, v in sa.items():
                score[k] += v
            for k, v in ua.items():
                staup[k] += v
            for k, v in wa.items():
                wpn[k] += v
            total[mode] += fr

    ids = sorted(set(seen["nothrow"]) | set(seen["throw"]))
    print(f"采样 {total['nothrow']:,} 帧(不投掷) + {total['throw']:,} 帧(投掷)")
    print()
    hdr = ("id  帧数    移动%  平均y   平均dx  平均|dx| 平均dy 平均|速| "
           "死亡归因 加分归因 回体力 得武器 | 投掷时帧数/不投掷时")
    print(hdr)
    print("-" * len(hdr))
    for oid in ids:
        st = stats[oid]
        n = st[0]
        if n == 0:
            continue
        nt, th = seen["nothrow"].get(oid, 0), seen["throw"].get(oid, 0)
        ratio = f"{th}/{nt}" if nt else f"{th}/0"
        print(f"{oid:3d} {int(n):7d} {100*st[1]/n:6.1f} {st[2]/n:7.1f} "
              f"{st[3]/n:8.1f} {st[4]/n:8.1f} {st[5]/n:7.1f} {st[6]/n:8.1f} "
              f"{death.get(oid,0):8d} {score.get(oid,0):8d} "
              f"{staup.get(oid,0):6d} {wpn.get(oid,0):6d} | {ratio}")

    print()
    print("判读要点:")
    print("  * 死亡归因高 = 真敌人（碰到掉命，不会自发发生）")
    print("  * 加分/回体力/得武器归因高 = 可拾取物（水果/道具）")
    print("  * 投掷时帧数 >> 不投掷时 = 玩家自己发出的子弹")
    print("  * 平均|速| 很高且 dx 单向 = 飞行物")


if __name__ == "__main__":
    main()
