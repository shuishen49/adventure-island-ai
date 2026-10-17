"""Is id 199 the fairy, and where is the invincibility timer?  (v4)

v1-v3 all failed to find anything, and each failure is informative:

  * v1 "the fairy vanished inside a 22x26 box" fired 4 times in 4
    consecutive decisions -- the sprite merely drifted out of the box.
  * v2/v3 TELEPORTED the player onto the sprite and then demanded the slot
    go empty.  8000 decisions, zero consumptions: id 199 is NOT eaten by
    walking into it (or the teleport does not move the player -- v4 checks
    that first).

So v4 stops assuming and instead records a movie: 8000 decisions of the
whole player-state region ($0060..$00FF) plus, every frame, how far the
fairy is.  Afterwards it looks for the byte whose behaviour matches
"invincibility": zero for a while, jumps up at/after a fairy contact, then
counts down to zero within a few seconds.

It also watches slot 0's OBJECT ID: the classic NES invincibility trick is
to blank the sprite by writing 0 into its slot, so any 64 -> 0 flicker is
strong evidence.

    python tools/probe_invincible4.py 1-3 > outputs/probe_inv4.log 2>&1
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
LO, HI = 0x0060, 0x0100          # player-state region to record

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

    # --- sanity: does writing slot 0 actually move the player? ---------
    x0 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes[R.OBJ.x_lo + ST * R.PLAYER_SLOT] = (
        nes[R.OBJ.x_lo + ST * R.PLAYER_SLOT] + 40) & 0xFF
    nes.controller = 0
    nes.step()
    x1 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    print(f"瞬移自检: 屏x {x0:.0f} -> {x1:.0f} "
          f"({'生效' if abs(x1 - x0) > 5 else '无效!'})")
    env.reset(seed=0)
    nes = env._nes

    N = 8000
    rec = np.zeros((N, HI - LO), dtype=np.uint8)
    slot0 = np.zeros(N, dtype=np.uint8)
    fd = np.full(N, 999.0, dtype=np.float32)     # fairy distance
    for t in range(N):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        raw = np.frombuffer(nes.save(), dtype=np.uint8)
        rec[t] = raw[LO:HI]
        slot0[t] = nes[OID + ST * R.PLAYER_SLOT]
        px_ = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        best = 999.0
        for s in range(1, 16):
            if int(nes[OID + ST * s]) == FAIRY_ID:
                dx = R.px(R.read_obj_world_x(nes, s)) - px_
                dy = R.px(R.read_obj_world_y(nes, s)) - py
                best = min(best, float(np.hypot(dx, dy)))
        fd[t] = best
        env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)
    env.close()

    print(f"id199 出现帧数: {(fd < 900).sum()} / {N}")
    near = np.nonzero(fd < 24)[0]
    print(f"距离 <24px 的帧: {near[:20]} ... 共 {len(near)}")

    # slot 0 id flicker?
    flick = np.nonzero(slot0 != R.PLAYER_ID)[0]
    print(f"slot0 id != {R.PLAYER_ID} 的帧数: {len(flick)}  "
          f"样例 {flick[:10]}")

    # --- look for the countdown ---------------------------------------
    if len(near) == 0:
        print("从未靠近 id199，无法定位计时器")
        return
    f0 = int(near[0])
    print(f"\n以第一次靠近 (帧 {f0}) 为界，扫描 ${LO:04X}..${HI - 1:04X}:")
    win = rec[max(0, f0 - 30): f0 + 500]
    pre = win[:30]
    post = win[30:]
    for i in range(HI - LO):
        a = LO + i
        if not (pre[:, i] == 0).all():
            continue
        v = post[:, i]
        nz = np.nonzero(v)[0]
        if len(nz) == 0 or nz[0] > 10:
            continue
        peak = int(v.max())
        last = int(nz[-1])
        decs = int(np.sum(np.diff(v[nz[0]:]) < 0))
        if peak < 4:
            continue
        kind = ("TIMER" if last < len(v) - 20 and decs >= 5 else "LATCH")
        print(f"  {kind} ${a:04X}: 起跳 {int(v[nz[0]])} 峰值 {peak} "
              f"下降 {decs} 次 归零于 +{last} 决策")


if __name__ == "__main__":
    main()
