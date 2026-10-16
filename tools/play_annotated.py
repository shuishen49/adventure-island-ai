"""Hold RIGHT + B (run/throw), tap JUMP at random, and record the
annotated frames -- a visual regression test for the object overlay.

Outputs:
  outputs/play_annotated.gif   the run, annotated, as an animated gif
  outputs/play_grid.png        a grid of stills (easy to eyeball)
  stdout                       per-frame stats: how many visible salmon
                               blobs (snails) got a box and how many did not

Usage:
  python tools/play_annotated.py [--frames N] [--sample K] [--seed S]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import (NES, NES_INPUT_A, NES_INPUT_B, NES_INPUT_RIGHT)  # noqa: E402
import ram_map as R                            # noqa: E402
from env import ROM                            # noqa: E402
from visualize import (annotate, boot, objects,  # noqa: E402
                       _ghost_blobs)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ppo/ppo_00092160.pt")
    ap.add_argument("--frames", type=int, default=7000)
    ap.add_argument("--sample", type=int, default=14,
                    help="keep 1 annotated frame every N game frames")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    nes = NES(ROM)
    boot(nes)
    start = nes.save()

    tr = R.SlotActivityTracker()

    frames: list[np.ndarray] = []
    enemy_frames: list[np.ndarray] = []   # stills that contain an enemy box
    jump_left = 0
    deaths = 0
    prev_lives = nes[R.LIVES]

    n_sample = 0
    n_with_enemy = 0
    n_with_fruit = 0
    blobs_total = 0
    blobs_boxed = 0
    unboxed_shots: list[np.ndarray] = []

    for i in range(args.frames):
        # --- input: hold right + B, tap A at random -------------------
        if jump_left == 0 and rng.random() < 0.035:
            jump_left = rng.randint(7, 12)
        ctrl = NES_INPUT_RIGHT | NES_INPUT_B
        if jump_left > 0:
            ctrl |= NES_INPUT_A
            jump_left -= 1
        name = "R+B" + ("+J" if jump_left > 0 else "")
        nes.controller = ctrl
        frame = np.asarray(nes.step(), dtype=np.uint8)

        objs = objects(nes)
        if i % args.sample == 0:
            shown = annotate(frame, nes, name, tr)
            n_sample += 1
            enemies = [o for o in objs
                       if o[1] in R.ENEMY_IDS
                       and not tr.is_stale(o[0])
                       and 0 <= o[2] < 256 and 0 <= o[3] < 240]
            fruits = [o for o in objs
                      if o[1] in R.FRUIT_IDS
                      and 0 <= o[2] < 256 and 0 <= o[3] < 240]
            n_with_enemy += bool(enemies)
            if enemies and len(enemy_frames) < 400:
                enemy_frames.append(shown)
            n_with_fruit += bool(fruits)

            # how many visible ground snails on screen got NO box at all?
            # a blob counts as covered if it sits on ANY drawn object
            # (enemy, fruit, item, checkpoint sign, ...), not just enemies.
            player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
            px, py = (player[2], player[3]) if player else (0, 0)
            drawn = [(o[2], o[3]) for o in objs
                     if -12 <= o[2] < 268 and -12 <= o[3] < 252]
            blobs = [b for b in _ghost_blobs(frame, [(px, py)])
                     if 140 <= b[1] <= 188 and b[2] >= 80 and abs(b[0] - px) > 12]
            missed = []
            for gx, gy, n in blobs:
                blobs_total += 1
                if any(abs(dx - gx) < 14 and abs(dy - gy) < 20
                       for dx, dy in drawn):
                    blobs_boxed += 1
                else:
                    missed.append((gx, gy))
            if missed and len(unboxed_shots) < 6:
                im = Image.fromarray(shown)
                d = ImageDraw.Draw(im)
                for gx, gy in missed:
                    d.rectangle([gx - 12, gy - 15, gx + 12, gy + 5],
                                outline=(255, 255, 0), width=1)
                    d.text((gx - 14, gy - 24), "NOBOX", fill=(255, 255, 0))
                unboxed_shots.append(np.asarray(im))

            frames.append(shown)
        else:
            tr.update(nes, [o[0] for o in objs])

        if nes[R.LIVES] < prev_lives:
            deaths += 1
        prev_lives = nes[R.LIVES]
        if nes[R.LIVES] == 0:
            nes.load(start)
            tr.reset()
            prev_lives = nes[R.LIVES]

    nes.controller = 0
    del nes

    # --- write gif ---------------------------------------------------
    gif_path = os.path.join(OUT, "play_annotated.gif")
    if frames:
        ims = [Image.fromarray(f).resize((512, 480), Image.NEAREST)
               for f in frames]
        ims[0].save(gif_path, save_all=True, append_images=ims[1:],
                    duration=90, loop=0, optimize=False)
    print(f"gif: {gif_path}  ({len(frames)} frames)")

    # --- grid of stills: prefer frames that actually contain an enemy -
    grid_src = enemy_frames if len(enemy_frames) >= 12 else frames
    picks = grid_src[:12]
    if picks:
        gw, gh = 4, 3
        cell = (256, 240)
        grid = Image.new("RGB", (cell[0] * gw, cell[1] * gh), (20, 20, 20))
        for k, f in enumerate(picks[: gw * gh]):
            grid.paste(Image.fromarray(f),
                       ((k % gw) * cell[0], (k // gw) * cell[1]))
        grid_path = os.path.join(OUT, "play_grid.png")
        grid.resize((cell[0] * gw * 2, cell[1] * gh * 2),
                    Image.NEAREST).save(grid_path)
        print(f"grid: {grid_path}  (source: "
              f"{'enemy frames' if grid_src is enemy_frames else 'all frames'}, "
              f"{len(enemy_frames)} enemy frames available)")

    for k, shot in enumerate(unboxed_shots):
        Image.fromarray(shot).resize((768, 720), Image.NEAREST).save(
            os.path.join(OUT, f"play_nobox_{k}.png"))

    print(f"--- sampled {n_sample} frames, deaths {deaths}")
    print(f"    frames with an enemy box : {n_with_enemy}")
    print(f"    frames with a fruit box  : {n_with_fruit}")
    print(f"    visible ground snails    : {blobs_total}, "
          f"boxed {blobs_boxed} "
          f"({100 * blobs_boxed / max(1, blobs_total):.1f}%), "
          f"NOBOX {blobs_total - blobs_boxed}")


if __name__ == "__main__":
    main()
