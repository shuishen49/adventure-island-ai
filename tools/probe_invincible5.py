"""Last sweep for the invincibility timer: record the WHOLE 2 KB.  (v5)

v4 narrowed the search to $0060..$00FF and found nothing, so the timer
(if it exists) lives somewhere else -- most likely in the object-slot
field arrays ($0564+) or in the OAM/sprite area.

v5 records every byte for 300 decisions around the first fairy contact and
reports every address that was 0 for 20 decisions before the contact and
non-zero afterwards, with a short trace so a countdown is obvious.

    python tools/probe_invincible5.py 1-3 > outputs/probe_inv5.log 2>&1
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


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    area, rnd = (int(p) for p in name.split("-"))
    env = AdventureIslandEnv(start_state=f"saves/stages/stage_{name}.state",
                             start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    ring = []          # rolling full-RAM snapshots
    contact = None
    for t in range(1200):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        ring.append(np.frombuffer(nes.save(), dtype=np.uint8).copy())
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
    print(f"接触点 @ 决策 {contact}")

    post = [ring[-1]]
    for _ in range(AFTER):
        nes.controller = 0
        nes.step()
        post.append(np.frombuffer(nes.save(), dtype=np.uint8).copy())
    allr = np.stack(ring + post)
    pre = allr[:BEFORE]
    tail = allr[BEFORE:]
    print(f"扫描 {allr.shape[1]} 个地址，窗口 {BEFORE} 决策前 / {AFTER} 决策后")
    hits = 0
    for a in range(allr.shape[1]):
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
              f"下降 {decs}  末值位置 +{last}  轨迹 {show}")
        hits += 1
        if hits > 60:
            print("  ...(截断)")
            break
    print(f"共 {hits} 个候选")
    print(f"参考: 体力 $0076 轨迹 {[int(x) for x in allr[:, R.STAMINA][::40]]}")


if __name__ == "__main__":
    main()
