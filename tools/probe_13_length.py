"""How long is 1-3 really, and what does the last stretch look like?

The "1-3 is 6900 px" number came from ONE god-mode script run.  Before we
declare the policy short of the goal we have to be sure the goal is where
we think it is -- if the flag really trips at, say, 6600 px then the
6607 px episode WAS a clear and every "0/12" since then is a measurement
artifact.

God mode (clearing $0072 every decision) is used only to remove DEATH from
the picture; the stage-complete flag $007A is never touched.

    python tools/probe_13_length.py            # 3 seeds, dumps the last frames
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import ACTIONS, AdventureIslandEnv                # noqa: E402

A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]

SHOT_AT = (6000, 6400, 6600, 6800)


def run(seed: int, dec: int = 5000):
    env = AdventureIslandEnv(start_state="saves/stages/stage_1-3.state",
                             start_with_axe=True, stage_index=2)
    env.reset(seed=seed)
    nes = env._nes
    best = 0.0
    shots: dict[int, str] = {}
    clear_at = None
    for t in range(dec):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        p = env._cam.progress(nes)
        if p > best:
            best = p
            for mark in SHOT_AT:
                if mark in shots:
                    continue
                if p >= mark:
                    arr = np.asarray(nes.step(), dtype=np.uint8).copy()
                    img = Image.fromarray(arr).convert("RGB").resize(
                        (1024, 960), Image.NEAREST)
                    path = os.path.join(_ROOT, "outputs",
                                        f"1-3_at{mark}.png")
                    img.save(path)
                    shots[mark] = path
        if int(nes[R.STAGE_COMPLETE]) != 0:
            clear_at = (t, p)
            break
        act = A_RJ if t % 26 in (0, 1) else (A_RT if t % 11 == 0 else A_RIGHT)
        env.step(act)
    env.close()
    return best, clear_at, shots


def main() -> None:
    for seed in (0, 1, 2, 3):
        best, clear_at, shots = run(seed)
        if clear_at:
            print(f"seed{seed}: STAGE_COMPLETE at t={clear_at[0]} "
                  f"progress={clear_at[1]:.0f} px")
        else:
            print(f"seed{seed}: no clear, best {best:.0f} px")
        for mark, path in sorted(shots.items()):
            print(f"          shot {mark}px -> {path}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
