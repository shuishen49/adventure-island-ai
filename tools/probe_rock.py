"""Where is the ROCK that only the fireball can destroy?

Game lore (Wikipedia / neoseeker FAQ): the stone axe cannot break rocks
and rolling boulders -- only the magical fireball can.  If that is true
there must be an object id that

    * sits at ground level (dy ~ 0 as seen from the player),
    * does not move (vx ~ 0),
    * and survives an axe hit but disappears when a fireball touches it.

This tool finds the CANDIDATES first (a scripted "run right, jump, throw"
bot surveys every stage and logs, for each object id, how often it is
seen, where it sits and whether it moves), and prints the ids that look
like a static ground-level blocker.  The second half -- hit it with an
axe, then with a fireball -- is tools/probe_rock_kill.py.

    python tools/probe_rock.py            # all 32 stages, 2500 decisions
    python tools/probe_rock.py 1-1 1-2    # just these
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, AdventureIslandEnv  # noqa: E402

OBJ_ID = R.OBJ.id
STRIDE = R.OBJ.STRIDE
N_SLOTS = 16

A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]

# ids we already understand -- no need to re-report them
KNOWN = (
    R.ENEMY_IDS + R.FRUIT_IDS + R.ITEM_IDS + R.SELF_IDS
    + R.LETHAL_STATIC_IDS + R.STATIC_IDS + (R.PLAYER_ID,)
)


def signed(b: int) -> int:
    return b - 256 if b > 127 else b


def stage_names() -> list[str]:
    return [f"{a}-{r}" for a in range(1, 9) for r in range(1, 5)]


def survey(name: str, decisions: int = 2500) -> dict:
    area, rnd = (int(p) for p in name.split("-"))
    state = os.path.join(_ROOT, "saves", "stages", f"stage_{name}.state")
    env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    seen: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "dy": [], "dx": [], "vx": []})
    stalls: list[tuple[float, list[tuple[int, float, float, int]]]] = []
    prev = env._cam.progress(nes)
    stall_for = 0
    dead_at = 0

    for t in range(decisions):
        if t % 26 in (0, 1):
            act = A_RJ                      # hop over anything
        elif t % 40 == 0:
            act = A_RT                      # and keep swinging
        else:
            act = A_RIGHT
        env.step(act)

        prog = env._cam.progress(nes)
        if prog <= prev + 0.01:
            stall_for += 1
        else:
            stall_for = 0
        prev = prog

        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        px_ = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        near = []
        for s in range(1, N_SLOTS):
            oid = int(nes[OBJ_ID + STRIDE * s])
            if oid == 0:
                continue
            dx = R.px(R.read_obj_world_x(nes, s)) - px_
            dy = R.px(R.read_obj_world_y(nes, s)) - py
            vx = signed(int(nes[R.OBJ.spd_x + STRIDE * s]))
            d = seen[oid]
            d["n"] += 1
            if len(d["dy"]) < 400:
                d["dy"].append(dy)
                d["dx"].append(dx)
                d["vx"].append(vx)
            if -8 < dx < 48 and abs(dy) < 32:
                near.append((oid, dx, dy, vx))
        if stall_for > 60:
            stalls.append((prog, near))
            stall_for = 0

        if int(nes[R.DEATH]) != 0:
            dead_at = t
            env.reset(seed=t)
            nes = env._nes
            prev = env._cam.progress(nes)
            stall_for = 0

    env.close()
    return {"name": name, "seen": dict(seen), "stalls": stalls,
            "dead_at": dead_at}


def main() -> None:
    args = sys.argv[1:] or stage_names()
    print(f"surveying {len(args)} stages, scripted run-right bot\n")
    for name in args:
        try:
            res = survey(name)
        except Exception as e:                       # noqa: BLE001
            print(f"{name}: FAILED {e}")
            continue
        seen = res["seen"]
        cands = []
        for oid, d in sorted(seen.items(), key=lambda kv: -kv[1]["n"]):
            if oid in KNOWN and oid not in R.OBSTACLE_IDS:
                continue
            dy = np.array(d["dy"]) if d["dy"] else np.zeros(1)
            vx = np.array(d["vx"]) if d["vx"] else np.zeros(1)
            ground = float(np.mean(np.abs(dy) < 26))
            still = float(np.mean(np.abs(vx) < 2))
            if d["n"] < 20:
                continue
            if ground > 0.7 and still > 0.7:
                cands.append((oid, d["n"], ground, still,
                              float(np.median(dy)), float(np.median(vx))))
        print(f"== {name} (died@{res['dead_at']}, "
              f"{len(res['stalls'])} stalls)")
        if not cands:
            print("   no new static ground-level id")
        for oid, n, g, s_, mdy, mvx in cands:
            lbl = R.kind_of(oid)[0]
            print(f"   id {oid:3d} [{lbl:9s}] n={n:5d} ground={g:.2f} "
                  f"still={s_:.2f} dy~{mdy:+.0f} vx~{mvx:+.1f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
