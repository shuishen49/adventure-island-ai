"""Does saves/stages/stage_1-4arena.state really restore the boss arena?

make_arena_state.py reported "重载后 progress = 0", which would make the
whole boss-practice idea useless.  Two suspects:

  1. the save file itself (load it into a bare NES -- no env involved)
  2. the env doing something after load (warm-up, stage pool, boot)

    python tools/check_arena_state.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES                                      # noqa: E402
import ram_map as R                                        # noqa: E402
from env import AdventureIslandEnv, ROM                    # noqa: E402

P = os.path.join(_ROOT, "saves", "stages", "stage_1-4arena.state")


def dump(tag, nes, cam=None):
    x = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    pr = cam.progress(nes) if cam else float("nan")
    print(f"{tag:<22} camera={int(nes[R.CAMERA]):3d} "
          f"travel={getattr(cam, 'travel', float('nan')):.0f} "
          f"progress={pr:7.1f}  player=({x:6.1f},{y:6.1f})  "
          f"$0011={int(nes[0x0011])}  sta={int(nes[R.STAMINA])}  "
          f"death={int(nes[R.DEATH])}  lives={int(nes[R.LIVES])}")


def main() -> None:
    raw = open(P, "rb").read()
    nes = NES(ROM)
    nes.load(np.frombuffer(raw, dtype=np.uint8).copy())
    print("1) bare NES, state loaded straight from disk:")
    print(f"   camera={int(nes[R.CAMERA])}  player_x="
          f"{R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)):.1f}  "
          f"$0011={int(nes[0x0011])}  sta={int(nes[R.STAMINA])}")

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3, start_state=P)
    cam = env._cam
    print("2) env built with that start_state, before reset:",
          "camera=", int(env._nes[R.CAMERA]))
    env.reset(seed=0)
    dump("3) after env.reset()", env._nes, cam)
    for _ in range(30):
        env.step(0)
    dump("4) after 30 no-ops", env._nes, cam)
    print("   info:", {k: v for k, v in
                       ("", {})}.__class__.__name__ if False else "")


if __name__ == "__main__":
    main()
