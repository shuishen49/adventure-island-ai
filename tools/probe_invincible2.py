"""Force-eat the id-199 fairy and diff ALL of RAM.  (v2)

v1 waited for the fairy to vanish inside a 22x26 box, but the sprite
drifts: four "events" fired on four consecutive decisions at the same
progress, i.e. it merely floated out of the box.  Nothing was eaten, so
no timer was found.

v2 teleports the player ONTO the fairy (the object-slot arrays are the
authoritative positions, so writing slot 0's x/y works), steps one frame,
and only accepts the event if the fairy's slot really went empty.  Then it
watches the whole 2 KB for 300 decisions and reports two signatures:

  A. COUNTDOWN -- 0 before, jumps up, then decreases and returns to 0
     (this is the invincibility timer).
  B. LATCH     -- 0 before, becomes non-zero and STAYS non-zero for the
     whole window (a state bit instead of a timer).

    python tools/probe_invincible2.py [stage]
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
PRE, POST = 8, 300

A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]


def snap(nes) -> np.ndarray:
    return np.frombuffer(nes.save(), dtype=np.uint8).copy()


def load(nes, raw: np.ndarray) -> None:
    nes.load(np.array(raw, dtype=np.uint8))     # must be writeable


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    area, rnd = (int(p) for p in name.split("-"))
    env = AdventureIslandEnv(start_state=f"saves/stages/stage_{name}.state",
                             start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    hist = []                       # rolling RAM snapshots
    events = 0
    for t in range(6000):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        hist.append(snap(nes))
        if len(hist) > PRE:
            hist.pop(0)

        slot = -1
        for s in range(1, 16):
            if int(nes[OID + ST * s]) == FAIRY_ID:
                sx = R.px(R.read_obj_world_x(nes, s))
                sy = R.px(R.read_obj_world_y(nes, s))
                if 16 < sx < 232 and 20 < sy < 210:
                    slot = s
                    break
        if slot < 0:
            env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)
            continue

        # --- teleport the player onto the fairy -----------------------
        before = snap(nes)
        for off, src in ((R.OBJ.x_hi, R.OBJ.x_hi), (R.OBJ.x_lo, R.OBJ.x_lo),
                         (R.OBJ.x_fr, R.OBJ.x_fr), (R.OBJ.y_lo, R.OBJ.y_lo),
                         (R.OBJ.y_fr, R.OBJ.y_fr)):
            nes[off + ST * R.PLAYER_SLOT] = nes[off + ST * slot]
        nes.controller = 0
        nes.step()
        eaten = all(int(nes[OID + ST * s]) != FAIRY_ID for s in range(16))
        if not eaten:
            load(nes, before)
            env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)
            continue

        events += 1
        print(f"\n=== 吃掉 id199 @ decision {t} "
              f"(progress {env._cam.progress(nes):.0f}px) ===")
        post = [snap(nes)]
        for _ in range(POST):
            nes.controller = 0
            nes.step()
            post.append(snap(nes))
        allr = np.stack(hist + post)
        col_pre = allr[:PRE]
        tail = allr[PRE:]
        for a in range(0x800):
            c = allr[:, a]
            if not (col_pre[:, a] == 0).all():
                continue
            v = tail[:, a]
            nz = np.nonzero(v)[0]
            if len(nz) == 0 or nz[0] > 8:
                continue
            peak = int(v.max())
            if peak < 4:
                continue
            last = nz[-1]
            decs = int(np.sum(np.diff(v[nz[0]:]) < 0))
            if last > POST - 30:                     # never came back to 0
                print(f"  LATCH ${a:04X}: 值 {int(v[nz[0]])} -> {peak} "
                      f"（{POST} 决策内不归零）")
            elif decs >= 5:
                print(f"  TIMER ${a:04X}: 起跳 {int(v[nz[0]])} 峰值 {peak} "
                      f"下降 {decs} 次 归零于第 {last} 决策")
        if events >= 3:
            break
        hist = []
        env.step(A_RIGHT)
    env.close()
    print(f"\n共触发 {events} 次吃精灵")


if __name__ == "__main__":
    main()
