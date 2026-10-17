"""Snapshot the screen the moment object id 199 is on stage, with a box.

    python tools/shot_fairy.py            # 1-3, 3 shots
Writes outputs/fairy_shot0.png ...
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

OID, ST = R.OBJ.id, R.OBJ.STRIDE
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]

WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 199
NAME = sys.argv[2] if len(sys.argv) > 2 else "1-3"
NSHOT = int(sys.argv[3]) if len(sys.argv) > 3 else 3
ZOOM = 4


def main() -> None:
    area, rnd = (int(p) for p in NAME.split("-"))
    state = os.path.join(_ROOT, "saves", "stages", f"stage_{NAME}.state")
    env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes
    shots = 0
    for t in range(4000):
        nes[R.STAMINA] = 10
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        act = A_RJ if t % 26 in (0, 1) else A_RIGHT
        env.step(act)
        frame = np.asarray(nes.step(), dtype=np.uint8).copy()
        slots = [s for s in range(0, 16) if int(nes[OID + ST * s]) == WANT]
        if not slots:
            continue
        img = Image.fromarray(frame).convert("RGB").resize(
            (256 * ZOOM, 240 * ZOOM), Image.NEAREST)
        dr = ImageDraw.Draw(img)
        for s in range(16):
            oid = int(nes[OID + ST * s])
            if oid == 0:
                continue
            sx = R.px(R.read_obj_world_x(nes, s))
            sy = R.px(R.read_obj_world_y(nes, s))
            sx = max(-40, min(296, sx))
            sy = max(-40, min(280, sy))
            col = (255, 60, 60) if oid == WANT else (90, 200, 255)
            dr.rectangle([(sx - 9) * ZOOM, (sy - 11) * ZOOM,
                          (sx + 9) * ZOOM, (sy + 9) * ZOOM],
                         outline=col, width=2 if oid == WANT else 1)
            dr.text(((sx - 9) * ZOOM, (sy - 21) * ZOOM), f"{oid}", fill=col)
        dr.text((6, 6), f"t={t}  id={WANT} slots={slots}", fill=(255, 255, 0))
        out = os.path.join(_ROOT, "outputs", f"fairy_shot{shots}.png")
        img.save(out)
        print("saved", out, "slots", slots)
        sys.stdout.flush()
        shots += 1
        if shots >= NSHOT:
            break
    if shots == 0:
        print("no sighting")
    env.close()


if __name__ == "__main__":
    main()
