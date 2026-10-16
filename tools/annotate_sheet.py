"""Headless annotated run: render the AI's view with all labels drawn,
then save a contact sheet + GIF so the annotations can be inspected.

    python tools/annotate_sheet.py runs/ppo/ppo_final.pt --frames 900
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cynes import NES, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, ROM  # noqa: E402
from enjoy import obs_tensor, push_img, reset_cam  # noqa: E402
from train import ActorCritic  # noqa: E402
from visualize import annotate, boot  # noqa: E402

OUT_DIR = os.path.join(ROOT, "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--manual", action="store_true",
                    help="drive with a fixed run-right+jump script instead "
                         "of a policy (no checkpoint needed)")
    args = ap.parse_args()

    net = None
    if args.checkpoint and not args.manual:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
        net.load_state_dict(ck["model"])
        net.eval()
        print(f"loaded {args.checkpoint} ({ck['steps']} steps)")
    else:
        print("manual script: run right + periodic jump")

    nes = NES(ROM)
    boot(nes)
    reset_cam(nes)
    cam = R.CameraTracker(nes)

    tiles, gifs, notes = [], [], []
    action, action_name = 0, "noop"
    prog0 = cam.progress(nes)
    for i in range(args.frames):
        if net is not None:
            if i % 4 == 0:
                with torch.no_grad():
                    action = int(net(obs_tensor(nes))[0].argmax())
                action_name = ACTIONS[action][1]
            ctrl = ACTIONS[action][0]
        else:
            from cynes import NES_INPUT_A, NES_INPUT_RIGHT
            ctrl = NES_INPUT_RIGHT | (NES_INPUT_A if (i % 40) < 12 else 0)
            action_name = "right+jump" if (i % 40) < 12 else "right"
        nes.controller = ctrl
        buf = np.asarray(nes.step(), dtype=np.uint8)
        push_img(nes, buf)
        # must be sampled EVERY frame: the scroll register wraps every 256px
        prog_now = cam.progress(nes) - prog0
        if i % args.every == 0:
            img = annotate(buf, nes, action_name)
            prog = prog_now
            notes.append((i, prog, nes[R.LIVES], nes[R.STAMINA]))
            tiles.append(Image.fromarray(img).resize((512, 480), Image.NEAREST))
            gifs.append(Image.fromarray(img).resize((384, 360), Image.NEAREST))
    nes.controller = 0
    del nes

    cols = len(tiles)
    sheet = Image.new("RGB", (512 * cols, 480), (0, 0, 0))
    for k, im in enumerate(tiles):
        sheet.paste(im, (k * 512, 0))
    sheet_path = os.path.join(OUT_DIR, "annotated_sheet.png")
    sheet.save(sheet_path)

    gif_path = os.path.join(OUT_DIR, "annotated.gif")
    gifs[0].save(gif_path, save_all=True, append_images=gifs[1:],
                 duration=120, loop=0)

    print("\nframe | progress_px | lives | stamina")
    for i, p, lv, st in notes:
        print(f"f{i:5d} | {p:10.1f} | {lv:5d} | {st:7d}")
    print(f"\nsheet -> {sheet_path}\ngif   -> {gif_path}")


if __name__ == "__main__":
    main()
