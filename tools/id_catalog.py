"""Build a visual catalogue: what every object id actually looks like.

Runs scripted play, and whenever an object id we have not sampled enough
yet shows up on screen, crops the pixels around it.  One row of patches
per id, labelled with the id and its current classification, so the
labelling can be checked by eye.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT, NES_INPUT_RIGHT

import ram_map as R
from env import ROM
from visualize import boot

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")
PATCH, SCALE = 48, 2                     # source crop size, upscale factor
PER_ID = 4                               # samples per id
CELL = PATCH * SCALE
OX, OY = 16, 10                          # object origin inside the crop


def main():
    nes = NES(ROM)
    boot(nes)

    samples: dict[int, list[np.ndarray]] = {}
    meta: dict[int, list[str]] = {}
    last_frame: dict[int, int] = {}

    for f in range(26000):
        # rich input mix so weapons, fruits, enemies and projectiles all show
        ctrl = NES_INPUT_RIGHT
        if (f % 37) < 10:
            ctrl |= NES_INPUT_A
        if (f % 53) < 6:
            ctrl |= NES_INPUT_B
        if (f % 401) < 40:
            ctrl = NES_INPUT_LEFT
        nes.controller = ctrl
        frame = np.asarray(nes.step(), dtype=np.uint8)
        if f % 3:
            continue

        px, py = (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
                  R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))
        for s in range(16):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid == 0:
                continue
            x = R.px(R.read_obj_world_x(nes, s))
            y = R.px(R.read_obj_world_y(nes, s))
            if not (OX + 4 <= x < 250 - OX and OY + 4 <= y < 230 - OY):
                continue
            got = samples.get(oid, [])
            if len(got) >= PER_ID:
                continue
            if f - last_frame.get(oid, -999) < 90:
                continue
            x0, y0 = int(x) - OX, int(y) - OY
            patch = frame[y0:y0 + PATCH, x0:x0 + PATCH].copy()
            if patch.shape[0] != PATCH or patch.shape[1] != PATCH:
                continue
            # crosshair on the object's reported origin
            patch[OY, OX - 6:OX - 2] = (255, 255, 255)
            patch[OY, OX + 3:OX + 7] = (255, 255, 255)
            patch[OY - 6:OY - 2, OX] = (255, 255, 255)
            patch[OY + 3:OY + 7, OX] = (255, 255, 255)
            samples.setdefault(oid, []).append(patch)
            last_frame[oid] = f
            spd = nes[R.OBJ.spd_x + R.OBJ.STRIDE * s]
            spd = spd - 256 if spd > 127 else spd
            meta.setdefault(oid, []).append(
                f"y{y:.0f} v{spd} dx{x - px:+.0f}")

    nes.controller = 0
    del nes

    ids = sorted(samples)
    cols = PER_ID + 1                      # label column + patches
    rows = len(ids)
    sheet = Image.new("RGB", (cols * CELL + 250, rows * CELL + 30), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), "Object id catalogue  -  id | label | sample patches "
                   "(what it looks like in game)", fill=(235, 235, 235))

    for r, oid in enumerate(ids):
        label, colour = R.kind_of(oid)
        ty = 30 + r * CELL
        d.text((8, ty + CELL // 2 - 6), f"id {oid}", fill=(255, 255, 255))
        d.text((70, ty + CELL // 2 - 6), label, fill=colour)
        for c, patch in enumerate(samples[oid]):
            im = Image.fromarray(patch).resize(
                (CELL, CELL), Image.NEAREST)
            sheet.paste(im, (250 + c * CELL, ty))
        note = "  ".join(meta[oid])
        d.text((8, ty + CELL // 2 + 10), note[:34], fill=(150, 150, 160))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "id_catalog.png")
    sheet.save(path)
    print(f"catalogue -> {path}   ({len(ids)} ids)")
    print()
    print("id   分类      样本数  备注")
    for oid in ids:
        label, _ = R.kind_of(oid)
        print(f"{oid:3d}  {label:8s}  {len(samples[oid])}     "
              f"{' | '.join(meta[oid])}")


if __name__ == "__main__":
    main()
