"""Does the FIREBALL actually open up ground the axe cannot?  (v2)

v1 was inconclusive for two reasons:
  * the scripted bot swung the axe on a fixed cadence, and throwing
    interrupts the run -- it never got past ~500 px;
  * a single trajectory per weapon is pure noise (on 4-3 the fireball run
    happened to do 861 px WORSE just because a different snail died).

v2 fixes both:
  * GOD MODE -- zeroing $0072 every decision cancels the death routine
    (measured: the same bot reaches 2095 px instead of 440 px on 3-1);
  * the bot only throws when something is actually in front of it, so the
    weapon is the ONLY difference between the two runs;
  * every (stage, weapon) pair is repeated over several seeds and the mean
    best-progress is compared.

If rocks really are axe-proof, some stage must show the fireball run
getting much further than the axe run.

    python tools/probe_fire_power.py                # default 3-1 / 4-3
    python tools/probe_fire_power.py --all
"""
from __future__ import annotations

import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R  # noqa: E402
from env import ACTIONS, AdventureIslandEnv  # noqa: E402

OBJ_ID = R.OBJ.id
STRIDE = R.OBJ.STRIDE

A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]

DECISIONS = 800
SEEDS = (0, 7, 21)


def run(name: str, weapon: int, seed: int, decisions: int = DECISIONS):
    area, rnd = (int(p) for p in name.split("-"))
    state = os.path.join(_ROOT, "saves", "stages", f"stage_{name}.state")
    env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=seed)
    nes = env._nes
    best = 0.0
    seen: Counter = Counter()
    for t in range(decisions):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0            # god mode
        nes[R.WEAPON] = weapon          # the cheat
        nes[R.STAMINA] = 10             # starvation is not what we test

        # --- the only decision: is something in front of me? -----------
        px_ = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        target = False
        for s in range(1, 16):
            oid = int(nes[OBJ_ID + STRIDE * s])
            if oid == 0:
                continue
            seen[oid] += 1
            dx = R.px(R.read_obj_world_x(nes, s)) - px_
            dy = R.px(R.read_obj_world_y(nes, s)) - py
            if 16 < dx < 220 and abs(dy) < 34 and oid != R.ID_PROJECTILE:
                target = True
        if target:
            act = A_RT                      # swing, but keep walking
        elif t % 26 in (0, 1):
            act = A_RJ                      # hop the gaps
        else:
            act = A_RIGHT
        env.step(act)
        best = max(best, env._cam.progress(nes))
    env.close()
    return best, seen


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    allst = "--all" in sys.argv[1:]
    stages = args or ([f"{a}-{r}" for a in range(1, 9) for r in range(1, 5)]
                      if allst else ["3-1", "4-3"])
    print(f"{len(stages)} stages x {len(SEEDS)} seeds, "
          f"{DECISIONS} decisions each\n")
    for name in stages:
        row = {}
        for w in (1, 2):
            vals = [run(name, w, s)[0] for s in SEEDS]
            row[w] = vals
        a, f = row[1], row[2]
        ma, mf = sum(a) / len(a), sum(f) / len(f)
        print(f"{name}  AXE {ma:7.0f}px {[round(v) for v in a]}"
              f"   FIRE {mf:7.0f}px {[round(v) for v in f]}"
              f"   diff {mf - ma:+7.0f}px")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
