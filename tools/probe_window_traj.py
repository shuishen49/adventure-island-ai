"""Replay the VISUALIZE loop exactly and log where the agent goes.

The window (visualize.py) does not use env.py -- it drives the emulator
itself: boot(), a persistent frame stack, a persistent CameraTracker,
one decision every 4 frames, auto-restart on $0072.  This script copies
that loop verbatim so we can see the trajectory the user is watching.

Reports, per decision: action, screen x (Px), world progress, scroll,
lives, death flag -- and flags any moment where progress moves BACKWARD
or where the game restarts.

    python tools/probe_window_traj.py runs/ppo_v3/ppo_00471040.pt
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

import ram_map as R
from cynes import NES
from env import ACTIONS, OBS_DIM
from enjoy import obs_tensor, push_img, reset_cam
from train import ActorCritic
from visualize import boot

DEATH_RESTART_DELAY = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--frames", type=int, default=6000)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    import env as _envmod
    nes = NES(_envmod.ROM)
    boot(nes)
    nes[R.WEAPON] = 1
    start_state = nes.save()
    reset_cam(nes)
    _b = np.asarray(nes.step(), dtype=np.uint8)
    for _ in range(2):
        push_img(nes, _b)

    cam = R.CameraTracker(nes)
    action, action_name = 0, "noop"
    death_countdown = None
    frame = 0
    restarts = 0
    last_restart_frame = 0
    prev_prog = cam.progress(nes)
    log = []

    while frame < a.frames:
        if frame % 4 == 0:
            with torch.no_grad():
                idx = int(net(obs_tensor(nes))[0].argmax())
            action = ACTIONS[idx][0]
            action_name = ACTIONS[idx][1]

        nes.controller = action
        buf = np.asarray(nes.step(), dtype=np.uint8)
        # decision-frame pushes only, mirroring the fixed visualize.py
        if frame % 4 == 3:
            push_img(nes, buf)
        frame += 1

        prog = cam.progress(nes)
        d = prog - prev_prog
        prev_prog = prog
        px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        log.append((frame, action_name, px, prog, d, nes[R.LIVES],
                    nes[R.DEATH], nes[R.CAMERA]))

        if nes[R.DEATH] != 0 and nes[R.STAGE_COMPLETE] == 0:
            if death_countdown is None:
                death_countdown = DEATH_RESTART_DELAY
        if death_countdown is not None:
            death_countdown -= 1
            if death_countdown <= 0:
                death_countdown = None
                print(f"  frame {frame:6d}: RESTART #{restarts + 1} "
                      f"(survived {frame - last_restart_frame} frames, "
                      f"progress was {prog:.0f})")
                nes.load(start_state)
                reset_cam(nes)
                _b = np.asarray(nes.step(), dtype=np.uint8)
                for _ in range(2):
                    push_img(nes, _b)
                cam = R.CameraTracker(nes)
                prev_prog = cam.progress(nes)
                restarts += 1
                last_restart_frame = frame

    # ---- report -------------------------------------------------------
    back = [(f, n, d) for f, n, _, _, d, *_ in log if d < 0]
    print(f"\n{len(log)} frames, {restarts} restarts, "
          f"{len(back)} backward frames "
          f"({sum(-d for _, _, d in back):.0f} px lost in total)")
    print("\nlargest backward jumps:")
    for f, n, d in sorted(back, key=lambda x: x[2])[:12]:
        print(f"  frame {f:6d}  {n:12s} {d:+8.1f} px")
    print("\ntrace every 200 frames (frame, action, Px, progress, lives, $72):")
    for f, n, px, prog, d, lv, dth, camv in log[::200]:
        print(f"  {f:6d} {n:12s} Px={px:6.1f} prog={prog:7.1f} "
              f"d={d:+6.1f} lives={lv} $72={dth:3d} scroll={camv}")


if __name__ == "__main__":
    main()
