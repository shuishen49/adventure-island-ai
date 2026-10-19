"""Capture annotated frames of the thing that kills the 1-4 runner (id 5).

The user names the cast from these pictures (7 = FROG, 10 = CROW,
32 = CHARGER, 56 = SPIDER), so: walk to the last stretch with the
invulnerability cheat, wait until id 5 is on screen, and write a few PNGs
with every object boxed and labelled "<name> id<id> dx,dy".

    python tools/snap_id5.py [--out outputs/id5]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,      # noqa: E402
                 NES_INPUT_A)

try:
    from PIL import Image, ImageDraw
except ImportError:                                        # pragma: no cover
    Image = None


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))


def annotate(frame: np.ndarray, nes, scale: int = 3) -> "Image.Image":
    img = Image.fromarray(frame).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    px, py = player_xy(nes)
    cam = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)) - 96.0   # screen x0
    for s in range(1, 16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        sx = (ox - (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)) - px)) \
            if False else (ox - cam)
        label, rgb = R.OBJ_KINDS.get(oid, ("?", (255, 255, 255)))
        x0, y0 = int(sx - 10) * scale, int(oy - 12) * scale
        x1, y1 = int(sx + 10) * scale, int(oy + 12) * scale
        d.rectangle([x0, y0, x1, y1], outline=rgb, width=2)
        d.text((x0, max(0, y0 - 12)), f"{label} id{oid} "
               f"dx{ox - px:+.0f} dy{oy - py:+.0f}", fill=rgb)
    d.rectangle([int(px - cam - 10) * scale, int(py - 12) * scale,
                 int(px - cam + 10) * scale, int(py + 12) * scale],
                outline=(60, 255, 60), width=2)
    d.text((4, 4), "PLAYER = green box", fill=(60, 255, 60))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "id5"))
    ap.add_argument("--shots", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam
    best, stall, shots = 0.0, 0, 0
    for f in range(60 * 60 * 10):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if f % 26 < 4 else 0)
        arr = np.asarray(nes.step(), dtype=np.uint8)
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        has5 = any(int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) == 5
                   for s in range(1, 16))
        if has5 and shots < args.shots and f % 40 == 0:
            img = annotate(arr, nes)
            path = os.path.join(args.out, f"id5_{shots + 1}.png")
            img.save(path)
            print(f"  {path}   progress={p:.0f} px", flush=True)
            shots += 1
        if shots >= args.shots:
            break
        if stall > 45 * 60:
            print(f"卡住 {best:.0f} px", flush=True)
            break
    print(f"共 {shots} 张, 目录 {args.out}")


if __name__ == "__main__":
    main()
