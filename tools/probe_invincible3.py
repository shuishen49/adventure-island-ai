"""Force-eat the id-199 sprite and dump EVERY byte that changed.  (v3)

v2 tried to guess the signature (a countdown / a latch) and printed
nothing at all -- which usually means the filter was wrong, not that
nothing happened.  v3 is deliberately dumb: at the moment the sprite is
consumed it records the whole 2 KB, and for every address whose value
changed it prints a compact trace

    $04AD  0 -> 24   [24, 21, 18, 12, 3, 0]

so a countdown is obvious by eye.  No "must be 0 before" filter, no
"must return to 0" filter.

    python tools/probe_invincible3.py [stage] > outputs/probe_inv3.log 2>&1
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
PROBES = (1, 4, 12, 30, 60, 120, 200)     # decisions after the pickup

A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]


def snap(nes) -> np.ndarray:
    return np.frombuffer(nes.save(), dtype=np.uint8).copy()


def load(nes, raw: np.ndarray) -> None:
    nes.load(np.array(raw, dtype=np.uint8))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    area, rnd = (int(p) for p in name.split("-"))
    env = AdventureIslandEnv(start_state=f"saves/stages/stage_{name}.state",
                             start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    events = 0
    first_seen = None
    for t in range(8000):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10

        slot = -1
        for s in range(1, 16):
            if int(nes[OID + ST * s]) == FAIRY_ID:
                sx = R.px(R.read_obj_world_x(nes, s))
                sy = R.px(R.read_obj_world_y(nes, s))
                if events == 0 and first_seen is None:
                    first_seen = (t, round(sx), round(sy))
                    print(f"首次见到 id199 @ 决策 {t}  屏坐标 ({sx:.0f},{sy:.0f})")
                if 4 < sx < 250 and 4 < sy < 235:
                    slot = s
                    break
        if slot < 0:
            env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)
            continue

        before = snap(nes)
        for off in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr,
                    R.OBJ.y_lo, R.OBJ.y_fr):
            nes[off + ST * R.PLAYER_SLOT] = nes[off + ST * slot]
        nes.controller = 0
        nes.step()
        eaten = all(int(nes[OID + ST * s]) != FAIRY_ID for s in range(16))
        if not eaten:
            load(nes, before)
            env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)
            continue

        events += 1
        after = snap(nes)
        prog = env._cam.progress(nes)
        print(f"=== 吃掉 id199 @ 决策 {t}, progress {prog:.0f}px, "
              f"slot {slot} ===")
        traces = {p: None for p in PROBES}
        done = 0
        for k in range(max(PROBES) + 1):
            if k in traces and traces[k] is None:
                traces[k] = snap(nes)
            if k > max(PROBES):
                break
            nes.controller = 0
            nes.step()
            done = k
        for a in range(0x800):
            b, af = int(before[a]), int(after[a])
            if b == af:
                continue
            row = [int(traces[p][a]) for p in PROBES]
            if af > 240 or max(row) > 240:
                if not (0 < af < 30):
                    continue
            print(f"  ${a:04X}  {b:3d} -> {af:3d}   "
                  f"[{', '.join(str(v) for v in row)}]")
            sys.stdout.flush()
        if events >= 2:
            break
        load(nes, before)
        env.step(A_RIGHT)
    env.close()
    print(f"共触发 {events} 次")


if __name__ == "__main__":
    main()
