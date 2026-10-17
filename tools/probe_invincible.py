"""Find the INVINCIBILITY timer in RAM (the "fairy" pickup, id 199).

The user identified id 199 (previously labelled DECO) as the little fairy
that grants a few seconds of invincibility, and asks whether the agent
should be rewarded for grabbing it.  To do that we need the RAM state.

Method: play with a god-mode bot, snapshot ALL 2 KB of RAM every decision
into a rolling buffer, and whenever an id-199 slot vanishes right next to
the player, look for bytes that

    * were 0 for the 10 decisions before the pickup,
    * became > 0 at/after the pickup,
    * then decreased monotonically (allowing stalls) and reached 0 again
      within 400 decisions.

That signature is a countdown timer -- i.e. the invincibility clock.
Eggplant (continuous vitality drain) is deliberately out of scope.

    python tools/probe_invincible.py [stage]
"""
from __future__ import annotations

import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, AdventureIslandEnv  # noqa: E402

OID, ST = R.OBJ.id, R.OBJ.STRIDE
FAIRY_ID = 199
PRE = 10           # decisions of "must be 0 before" required
POST = 400         # decisions to watch the countdown

A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]


def snap(nes) -> np.ndarray:
    return np.frombuffer(nes.save(), dtype=np.uint8).copy()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    area, rnd = (int(p) for p in name.split("-"))
    env = AdventureIslandEnv(start_state=f"saves/stages/stage_{name}.state",
                             start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    buf: deque_t = []                     # (decision, ram) rolling
    events = 0
    seen_199 = Counter()
    for t in range(4000):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0              # god mode
        nes[R.STAMINA] = 10
        ram = snap(nes)
        buf.append((t, ram))
        if len(buf) > PRE + POST:
            buf.pop(0)

        # --- is the fairy on screen, and where? -----------------------
        px_ = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        fairy_here = False
        for s in range(1, 16):
            if int(nes[OID + ST * s]) == FAIRY_ID:
                seen_199[round(R.px(R.read_obj_world_x(nes, s)) - px_)] += 1
                if abs(R.px(R.read_obj_world_x(nes, s)) - px_) < 22 and \
                        abs(R.px(R.read_obj_world_y(nes, s)) - py) < 26:
                    fairy_here = True
        if not fairy_here:
            env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)
            continue

        # fairy within touch range: step one more decision, then analyse
        env.step(A_RIGHT)
        ram2 = snap(nes)
        buf.append((t + 1, ram2))
        gone = all(int(ram2[OID + ST * s]) != FAIRY_ID for s in range(16))
        if not gone:
            continue

        events += 1
        print(f"\n=== event {events} @ decision {t} "
              f"(progress {env._cam.progress(nes):.0f}px) ===")
        hist = np.stack([r for _, r in buf[-(PRE + POST):]])
        n = hist.shape[0]
        cands = []
        for a in range(0x800):
            col = hist[:, a]
            if not (col[:PRE] == 0).all():
                continue                       # must be 0 before pickup
            after = col[PRE:]
            nz = np.nonzero(after)[0]
            if len(nz) == 0 or nz[0] > 6:
                continue                       # must rise promptly
            peak = after[nz[0]:].max()
            if peak < 8:
                continue                       # too short to be a timer
            tail = after[nz[0]:]
            dec = np.sum(np.diff(tail) < 0)    # decreasing steps
            if dec < 5:
                continue
            last = np.nonzero(tail)[0]
            if last[-1] > POST - 40:
                continue                       # must fall back to 0
            cands.append((a, int(tail[nz[0]]), int(tail.max()), dec,
                          int(last[-1])))
        if not cands:
            print("  no timer-like byte found")
        for a, v0, vmax, dec, life in sorted(cands, key=lambda c: -c[1]):
            print(f"  ${a:04X}: 起跳值 {v0:3d}  峰值 {vmax:3d}  "
                  f"下降步数 {dec:3d}  归零耗时 {life} 决策")
        if events >= 4:
            break
        buf = buf[-PRE:]
        env.step(A_RIGHT)
    env.close()
    if not events:
        print(f"never touched id {FAIRY_ID}; dx histogram {seen_199.most_common(8)}")


class deque_t(list):        # typing shim, list is enough here
    pass


if __name__ == "__main__":
    main()
