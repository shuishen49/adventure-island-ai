"""Is the god-mode STAGE_COMPLETE on 1-3 real, or an artefact of clearing $0072?

tools/probe_13_length.py saw $007A trip at 5074 / 6119 / 5331 / 5847 px --
four different answers for one stage, and all of them below the 6607 px our
policy has already reached without ever clearing.  Either the flag means
something else under god mode, or "6900 px" was never the finish line.

The only way to tell the two apart is to look at the screen at the moment
the flag goes up.

    python tools/probe_clear_frame.py 1-3
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


def run(stage: str, seed: int = 1, dec: int = 5000):
    area, rnd = (int(p) for p in stage.split("-"))
    env = AdventureIslandEnv(
        start_state=f"saves/stages/stage_{stage}.state",
        start_with_axe=True, stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=seed)
    nes = env._nes
    for t in range(dec):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        if int(nes[R.STAGE_COMPLETE]) != 0:
            prog = env._cam.progress(nes)
            print(f"seed{seed}: $007A up at t={t} progress={prog:.0f} px  "
                  f"AREA={int(nes[R.AREA])} ROUND={int(nes[R.ROUND])}")
            for k in range(240):          # 4 s of the clear animation
                arr = np.asarray(nes.step(), dtype=np.uint8).copy()
                if k in (0, 30, 90, 180):
                    img = Image.fromarray(arr).convert("RGB").resize(
                        (1024, 960), Image.NEAREST)
                    p = os.path.join(_ROOT, "outputs",
                                     f"clear_{stage}_f{k:03d}.png")
                    img.save(p)
                    print(f"          frame +{k:3d} -> {p}")
            env.close()
            return
        act = A_RJ if t % 26 in (0, 1) else (A_RT if t % 11 == 0 else A_RIGHT)
        env.step(act)
    print(f"seed{seed}: $007A never went up")
    env.close()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "1-3")
