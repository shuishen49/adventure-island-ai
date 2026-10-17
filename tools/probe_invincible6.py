"""Invincibility-timer probe, this time against REAL CPU RAM.  (v6)

Earlier versions indexed into ``nes.save()`` as if it were RAM.  It is
not: cynes' save blob is 5382 bytes (emulator state header + RAM + PPU +
APU ...), so addresses like $0CDB were offsets into that blob, i.e.
meaningless.  This version reads the 2 KB of CPU RAM through ``nes[i]``,
which is the same accessor the env itself uses, so the addresses are
real.

Method as before: play until an id-199 sprite is within 24 px of the
player, then record the whole 2 KB for 40 decisions before / 260 after,
and report every address that was 0 before and non-zero after, with a
trace (a countdown is obvious by eye).

    python tools/probe_invincible6.py 1-3 > outputs/probe_inv6.log 2>&1
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, AdventureIslandEnv  # noqa: E402

OID, ST = R.OBJ.id, R.OBJ.STRIDE
FAIRY_ID = 199
BEFORE, AFTER = 40, 260

A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]


def ram(nes) -> np.ndarray:
    return np.array([nes[i] for i in range(0x800)], dtype=np.uint8)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    area, rnd = (int(p) for p in name.split("-"))
    env = AdventureIslandEnv(start_state=f"saves/stages/stage_{name}.state",
                             start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    ring = []
    contact = None
    for t in range(1200):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        ring.append(ram(nes))
        if len(ring) > BEFORE:
            ring.pop(0)
        px_ = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        for s in range(1, 16):
            if int(nes[OID + ST * s]) != FAIRY_ID:
                continue
            d = float(np.hypot(R.px(R.read_obj_world_x(nes, s)) - px_,
                               R.px(R.read_obj_world_y(nes, s)) - py))
            if d < 24:
                contact = t
                break
        if contact is not None:
            break
        env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)

    if contact is None:
        print("没靠近过 id199")
        return
    print(f"接触点 @ 决策 {contact}，开始录 {AFTER} 决策 ...")

    post = [ring[-1]]
    for _ in range(AFTER):
        nes.controller = 0
        nes.step()
        post.append(ram(nes))
    allr = np.stack(ring + post)
    pre, tail = allr[:BEFORE], allr[BEFORE:]
    print(f"扫描真实 CPU RAM 2048 字节，前 {BEFORE} / 后 {AFTER} 决策")

    hits = 0
    for a in range(0x800):
        if not (pre[:, a] == 0).all():
            continue
        v = tail[:, a]
        nz = np.nonzero(v)[0]
        if len(nz) == 0 or nz[0] > 12:
            continue
        peak = int(v.max())
        last = int(nz[-1])
        decs = int(np.sum(np.diff(v[nz[0]:]) < 0))
        show = [int(v[k]) for k in (nz[0], 10, 30, 60, 120, min(last, 250))]
        print(f"  ${a:04X}  0 -> {int(v[nz[0]])}  峰值 {peak}  "
              f"下降 {decs}  末值 +{last}  轨迹 {show}")
        hits += 1
    print(f"共 {hits} 个候选")
    print(f"参考 $0076 体力: {[int(x) for x in allr[:, R.STAMINA][::40]]}")
    print(f"参考 $003D/3E/3F: "
          f"{[int(allr[0, a]) for a in (0x3D, 0x3E, 0x3F)]}")


if __name__ == "__main__":
    main()
