"""Capture annotated frames of unknown / suspicious object ids.

Generalised from tools/snap_id5.py.  The user names the cast from these
pictures (7=FROG, 10=CROW, 32=CHARGER, 56=SPIDER, 67/198=boss), so this
walks the stage with the invulnerability cheat and writes a few PNGs with
every object boxed and labelled "<name> id<id> dx,dy".

    python tools/snap_unknown.py --round 4 --stage-idx 4 --ids 36,19 \
        --out outputs/id36 --shots 4

NOTE: the cheat (DEATH=0 / STAMINA=10 every frame) is only used to *reach*
the object so it can be photographed.  Never trust a "stage length" or
"stage clear" measured under this cheat -- see the 1-4 notes in
历史记录/1-2到1-4训练挫折史.md (坑 18).
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
from env import AdventureIslandEnv, NES_INPUT_A, NES_INPUT_RIGHT  # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:                                        # pragma: no cover
    Image = None


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))


def annotate(frame: np.ndarray, nes, scale: int = 4) -> "Image.Image":
    img = Image.fromarray(frame).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    px, py = player_xy(nes)
    cam = px - 96.0                       # left edge of the screen in px
    for s in range(1, 16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        sx = ox - cam
        label, rgb = R.OBJ_KINDS.get(oid, ("?", (255, 255, 255)))
        x0, y0 = int(sx - 10) * scale, int(oy - 12) * scale
        x1, y1 = int(sx + 10) * scale, int(oy + 12) * scale
        d.rectangle([x0, y0, x1, y1], outline=rgb, width=2)
        d.text((x0, max(0, y0 - 14)), f"{label} id{oid} "
               f"dx{ox - px:+.0f} dy{oy - py:+.0f}", fill=rgb)
    d.rectangle([int(px - cam - 10) * scale, int(py - 12) * scale,
                 int(px - cam + 10) * scale, int(py + 12) * scale],
                outline=(60, 255, 60), width=2)
    d.text((4, 4), "PLAYER = green box", fill=(60, 255, 60))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=4,
                    help="0-based $0038 (3 = 1-4, 4 = 2-1)")
    ap.add_argument("--stage-idx", type=int, default=4)
    ap.add_argument("--ids", default="36",
                    help="comma separated object ids to photograph")
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "unknown"))
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    want = {int(x) for x in args.ids.split(",") if x.strip()}
    os.makedirs(args.out, exist_ok=True)

    env = AdventureIslandEnv(start_with_axe=True, start_round=args.round,
                             stage_index=args.stage_idx)
    env.reset(seed=0)
    nes, cam = env._nes, env._cam
    best, stall, shots = 0.0, 0, 0
    for f in range(60 * 60 * 12):
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
        onscreen = {int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
                    for s in range(1, 16)}
        if (onscreen & want) and shots < args.shots and f % 30 == 0:
            img = annotate(arr, nes, args.scale)
            path = os.path.join(
                args.out, f"id{'_'.join(str(i) for i in sorted(want))}"
                          f"_{shots + 1}.png")
            img.save(path)
            print(f"  {path}   progress={p:.0f} px  ids={sorted(onscreen)}",
                  flush=True)
            shots += 1
        if shots >= args.shots:
            break
        if stall > 45 * 60:
            print(f"stalled at {best:.0f} px", flush=True)
            break
    print(f"{shots} shots -> {args.out}   (farthest {best:.0f} px)")


if __name__ == "__main__":
    main()
