"""Capture the exact frames where the policy dies.

Runs headless (same inference path as the watch window) and, at every
death, saves the frame just BEFORE the fatal one and the frame of the
death itself, annotated.  Shows what is actually killing the agent.

Usage:  python tools/probe_death_frames.py <ckpt> [--episodes N]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES                # noqa: E402
import ram_map as R                  # noqa: E402
from env import ACTIONS, OBS_DIM, ROM  # noqa: E402
from enjoy import obs_tensor, push_img, reset_cam  # noqa: E402
from train import ActorCritic        # noqa: E402
from visualize import annotate, boot, objects  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=60000)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    boot(nes)
    start = nes.save()
    reset_cam(nes)
    tr = R.SlotActivityTracker()
    cam = R.CameraTracker(nes)

    action = 0
    prev_frame: np.ndarray | None = None
    prev_x = cam.progress(nes)
    prev_y = 0.0
    prev_lives = nes[R.LIVES]
    saved = 0
    runs = 0
    seen_spots: set[int] = set()

    for i in range(args.max_frames):
        if i % 4 == 0:
            with torch.no_grad():
                action = int(net(obs_tensor(nes))[0].argmax())
        nes.controller = ACTIONS[action][0]
        frame = np.asarray(nes.step(), dtype=np.uint8)
        push_img(nes, frame)

        py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        prog = cam.progress(nes)

        lv = nes[R.LIVES]
        if lv < prev_lives and prev_frame is not None and saved < 6:
            spot = int(prev_x) // 32 * 32
            if spot not in seen_spots:
                seen_spots.add(spot)
                name = ACTIONS[action][1]
                before = annotate(prev_frame, nes, "before:" + name, tr)
                now = annotate(frame, nes, "DEATH:" + name, tr)
                both = np.concatenate([before, now], axis=1)
                Image.fromarray(both).resize((1024, 480),
                                             Image.NEAREST).save(
                    os.path.join(OUT, f"deathspot_{saved}.png"))
                print(f"f{i}: death #{len(seen_spots)} at x={prev_x:.0f} "
                      f"(screen y={prev_y:.0f}, act={name}) "
                      f"-> outputs/deathspot_{saved}.png")
                saved += 1
            prev_lives = lv
            if lv == 0:
                runs += 1
                if runs >= args.episodes:
                    break
                nes.load(start)
                reset_cam(nes)
                tr.reset()
                cam = R.CameraTracker(nes)
                prev_lives = nes[R.LIVES]
                prev_frame = None
                prev_x = cam.progress(nes)
                continue
        else:
            prev_lives = lv

        prev_frame = frame
        prev_y = py
        prev_x = prog

    nes.controller = 0
    del nes
    print(f"--- captured {saved} distinct death spots")


if __name__ == "__main__":
    main()
