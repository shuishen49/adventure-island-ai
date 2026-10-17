"""Take a zoomed, boxed screenshot of one object id on one stage.

    python tools/probe_rock_shot.py 63 3-1
Writes outputs/id63_3-1_shot0.png ... (up to --n shots).

The bot is the usual "run right, hop, swing" one; stamina is topped up so
it does not starve before it meets the object.  Screen coordinates come
straight from the object slots (they are SCREEN-space, see ram_map.py).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

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

ZOOM = 5
PAD = 26


def main() -> None:
    oid_want = int(sys.argv[1])
    name = sys.argv[2]
    want_n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    area, rnd = (int(p) for p in name.split("-"))
    state = os.path.join(_ROOT, "saves", "stages", f"stage_{name}.state")
    env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    shots = 0
    for t in range(6000):
        nes[R.STAMINA] = 10
        if int(nes[R.DEATH]) != 0:      # god mode, see probe_fire_power.py
            nes[R.DEATH] = 0
        act = A_RJ if t % 26 in (0, 1) else (A_RT if t % 13 == 0 else A_RIGHT)
        env.step(act)
        buf = np.asarray(nes.step(), dtype=np.uint8).copy()
        for s in range(1, 16):
            if int(nes[OBJ_ID + STRIDE * s]) != oid_want:
                continue
            sx = R.px(R.read_obj_world_x(nes, s))
            sy = R.px(R.read_obj_world_y(nes, s))
            if not (6 < sx < 240 and 6 < sy < 225):
                continue
            img = Image.fromarray(buf, mode="RGB")
            d = ImageDraw.Draw(img)
            d.rectangle([sx - PAD, sy - PAD, sx + PAD, sy + PAD],
                        outline=(255, 60, 60), width=1)
            x0, y0 = max(0, int(sx) - 60), max(0, int(sy) - 45)
            crop = img.crop((x0, y0, min(256, x0 + 120), min(240, y0 + 90)))
            crop = crop.resize((crop.width * ZOOM, crop.height * ZOOM),
                               Image.NEAREST)
            out = os.path.join(_ROOT, "outputs",
                               f"id{oid_want}_{name}_shot{shots}.png")
            crop.save(out)
            print(f"wrote {out}  (screen x={sx:.0f} y={sy:.0f}, "
                  f"decision {t})")
            shots += 1
            if shots >= want_n:
                env.close()
                return
        if int(nes[R.DEATH]) != 0:
            env.reset(seed=t)
            nes = env._nes
    env.close()
    print(f"never saw id {oid_want} on {name}")


if __name__ == "__main__":
    main()
