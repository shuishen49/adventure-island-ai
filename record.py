"""Record an episode played by a trained checkpoint into a GIF + contact sheet.

    python record.py runs/ppo/ppo_00081920.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from cynes import NES, NES_INPUT_START

import ram_map as R
from env import ACTIONS, OBS_DIM
from enjoy import obs_tensor, push_img, reset_cam
from train import ActorCritic

import env as _envmod
ROM = _envmod.ROM
OUT = "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_final.pt")
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--every", type=int, default=12,
                    help="capture 1 frame every N game frames")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"loaded {args.checkpoint} ({ck['steps']} env steps)")

    nes = NES(ROM)
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=60)
    start_lives = nes[R.LIVES]
    print("booted into gameplay, lives =", start_lives)

    shots, action = [], 0
    max_x, deaths = 0.0, 0
    for i in range(args.frames):
        if i % 4 == 0:
            with torch.no_grad():
                action = int(net(obs_tensor(nes))[0].argmax())
        nes.controller = ACTIONS[action][0]
        arr = np.asarray(nes.step(), dtype=np.uint8)
        push_img(nes, arr)

        x = R.px(R.read_obj_world_x(nes, 0))
        max_x = max(max_x, x)
        if i % args.every == 0:
            shots.append((arr.copy(), ACTIONS[action][1], i, x))

        if nes[R.LIVES] < start_lives:
            deaths += 1
            print(f"  died at frame {i}, x={x:.0f}px")
            break

    print(f"episode: {len(shots)} captures, max x = {max_x:.0f}px")

    # ---------- GIF ----------
    gif_frames = [
        Image.fromarray(a, mode="RGB").resize(
            (256 * args.scale, 240 * args.scale), Image.NEAREST)
        for a, _, _, _ in shots
    ]
    gif_path = os.path.join(OUT, "ai_episode.gif")
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=60, loop=0, optimize=True)
    print("gif  ->", gif_path, f"({os.path.getsize(gif_path) / 1e6:.1f} MB)")

    # ---------- contact sheet with labels ----------
    cols, cw, ch = 6, 256, 240
    rows = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * (ch + 18)), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for n, (a, act, fr, x) in enumerate(shots):
        r, c = divmod(n, cols)
        sheet.paste(Image.fromarray(a, mode="RGB"), (c * cw, r * (ch + 18)))
        d.text((c * cw + 3, r * (ch + 18) + 4),
               f"f{fr:<5} {act:<12} x={x:6.0f}px", fill=(20, 20, 20))
    sheet_path = os.path.join(OUT, "ai_episode_sheet.png")
    sheet.save(sheet_path)
    print("sheet ->", sheet_path)

    del nes


if __name__ == "__main__":
    main()
