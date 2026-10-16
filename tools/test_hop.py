"""Can any simple strategy get past the obstacle at x~96?

Tries several hand-coded policies and reports the furthest x reached.
If none get past it, the environment (or the ROM state after boot) is
broken and further training is pointless.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_A, NES_INPUT_LEFT, NES_INPUT_RIGHT  # noqa: E402
from cynes import NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = r"D:\source_code\cynes\冒险岛1.nes"


def boot(nes):
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def px_x(nes):
    return R.px(R.read_obj_world_x(nes, 0))


def px_y(nes):
    return R.px(R.read_obj_world_y(nes, 0))


def run_policy(name, policy_fn, frames=900):
    nes = NES(ROM)
    boot(nes)
    best_x, start_x = px_x(nes), px_x(nes)
    ys = []
    for i in range(frames):
        nes.controller = policy_fn(i)
        nes.step()
        x = px_x(nes)
        best_x = max(best_x, x)
        if i % 30 == 0:
            ys.append(int(px_y(nes)))
    gained = best_x - start_x
    print(f"  {name:<28} start={start_x:6.1f} best={best_x:7.1f} "
          f"gained={gained:+7.1f}px  y={ys[:10]}")
    del nes
    return gained


print("policies (900 frames each):")
run_policy("right only", lambda i: NES_INPUT_RIGHT)
run_policy("right + hop every 20f",
           lambda i: NES_INPUT_RIGHT | (NES_INPUT_A if i % 20 < 8 else 0))
run_policy("right + hop every 30f",
           lambda i: NES_INPUT_RIGHT | (NES_INPUT_A if i % 30 < 10 else 0))
run_policy("right + hold A always", lambda i: NES_INPUT_RIGHT | NES_INPUT_A)
run_policy("right + hop every 12f",
           lambda i: NES_INPUT_RIGHT | (NES_INPUT_A if i % 12 < 5 else 0))
run_policy("left only", lambda i: NES_INPUT_LEFT)
run_policy("left then right",
           lambda i: NES_INPUT_LEFT if i < 200 else NES_INPUT_RIGHT)

# Does the player actually fall / die?  Watch one episode closely.
print("\nclose watch: right + hop, first 240 frames")
nes = NES(ROM)
boot(nes)
for i in range(240):
    nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if i % 30 < 10 else 0)
    nes.step()
    if i % 20 == 0:
        print(f"  f{i:3d} x={px_x(nes):7.1f} y={px_y(nes):6.1f} "
              f"lives={nes[R.LIVES]} death={nes[R.DEATH]} sta={nes[R.STAMINA]}")
del nes
