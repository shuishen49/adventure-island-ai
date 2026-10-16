"""Probe: what happens to enemy object slots when the player dies?

User report: after dying, enemy boxes stay frozen at fixed screen spots
even where no enemy is visible.

This probe plays greedily with a checkpoint, waits for a death (LIVES
decrement), then logs the DEATH flag ($0072) and every object slot's
(id, x, y, vx) around the death, and saves RAW frames (no annotation)
so we can see whether a sprite is actually rendered at the slot position.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
from cynes import NES

import ram_map as R
from env import ACTIONS, ROM
from enjoy import obs_tensor, reset_cam
from train import ActorCritic
from visualize import boot, objects


def slots_str(nes):
    parts = []
    for s, oid, x, y, vx in objects(nes):
        parts.append(f"s{s}:id{oid}@({x:.0f},{y:.0f})v{vx}")
    return " ".join(parts) if parts else "(none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_00092160.pt")
    ap.add_argument("--max-frames", type=int, default=12000)
    ap.add_argument("--wait-after", type=int, default=400)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    import env as _env
    net = ActorCritic(_env.OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    boot(nes)
    reset_cam(nes)

    out_dir = os.path.join("outputs", "death_probe")
    os.makedirs(out_dir, exist_ok=True)

    last_lives = nes[R.LIVES]
    action = 0
    death_frame = None
    log = []
    for i in range(args.max_frames):
        if i % 4 == 0:
            with torch.no_grad():
                action = int(net(obs_tensor(nes))[0].argmax())
        nes.controller = ACTIONS[action][0]
        frame = np.asarray(nes.step(), dtype=np.uint8)
        lv = nes[R.LIVES]
        death_flag = nes[R.DEATH]

        if death_frame is None:
            if i % 30 == 0 or death_flag != 0:
                log.append(f"f{i:5d} lives={lv} DEATH={death_flag} {slots_str(nes)}")
            if lv < last_lives:
                last_lives = lv
                death_frame = i
                log.append(f"=== DEATH at f{i}  lives={lv} DEATH={death_flag} ===")
                Image.fromarray(frame).resize((768, 720), Image.NEAREST).save(
                    os.path.join(out_dir, "death_000.png"))
        else:
            n = i - death_frame
            if n % 10 == 0:
                log.append(f"f{i:5d} (+{n:3d}) lives={lv} DEATH={death_flag} "
                           f"{slots_str(nes)}")
            if n in (10, 40, 80, 120):
                Image.fromarray(frame).resize((768, 720), Image.NEAREST).save(
                    os.path.join(out_dir, f"death_{n:03d}.png"))
            if n >= args.wait_after:
                break

    nes.controller = 0
    print("\n".join(log))
    print(f"\nraw frames saved to {out_dir}/")
    del nes


if __name__ == "__main__":
    main()
