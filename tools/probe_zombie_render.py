"""Probe: does the game still RENDER a zombie slot's sprite after respawn?

Find a death where an enemy is ON-SCREEN, record its screen position,
then after respawn save raw frames and report whether snail-coloured
pixels still appear at that position.
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
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, reset_cam
from train import ActorCritic
from visualize import boot, objects

# snail body pink / shell magenta-ish (sampled from the sprite)
SNAIL_COLORS = [(231, 84, 128), (216, 40, 96), (247, 171, 198)]


def snail_at(frame, x, y, half=10):
    """True if snail-coloured pixels sit near (x, y)."""
    x0, x1 = max(0, int(x) - half), min(256, int(x) + half)
    y0, y1 = max(0, int(y) - 20), min(240, int(y) + 4)
    patch = frame[y0:y1, x0:x1].astype(int)
    for c in SNAIL_COLORS:
        d = np.abs(patch - np.array(c)).sum(axis=2)
        if (d < 90).any():
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_00092160.pt")
    ap.add_argument("--max-frames", type=int, default=15000)
    ap.add_argument("--random", action="store_true",
                    help="random policy (dies in more varied ways)")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    boot(nes)
    reset_cam(nes)
    rng = np.random.default_rng(0)

    out = os.path.join("outputs", "zombie_render")
    os.makedirs(out, exist_ok=True)

    action = 0
    prev_lives = nes[R.LIVES]
    death_at = None
    frozen_enemy = None      # (slot, x, y) of an onscreen enemy at death

    for i in range(args.max_frames):
        if i % 4 == 0:
            if args.random:
                action = int(rng.integers(len(ACTIONS)))
            else:
                with torch.no_grad():
                    action = int(net(obs_tensor(nes))[0].argmax())
            # charge mode: if an enemy is onscreen, run into it on purpose
            # so the death definitely happens with the enemy visible
            ens = [(s, x, y) for s, oid, x, y, v in objects(nes)
                   if oid in R.ENEMY_IDS and 0 <= x < 256 and 0 <= y < 240]
            if ens:
                frozen_enemy = ens[0]
                if death_at is None or death_at == -1:
                    ex = ens[0][1]
                    p_x = nes[R.OBJ.x_lo]
                    if abs(ex - p_x) < 130:
                        action = 1 if ex > p_x else 2      # right / left
        nes.controller = ACTIONS[action][0]
        frame = np.asarray(nes.step(), dtype=np.uint8)
        lv = nes[R.LIVES]

        if death_at is None or death_at < 0:
            if lv < prev_lives:
                prev_lives = lv
                if frozen_enemy is None:
                    if death_at is None:
                        print(f"f{i}: death but no onscreen enemy, "
                              f"waiting for next death")
                    death_at = -1
                    continue
                death_at = i
                print(f"f{i}: DEATH with onscreen enemy {frozen_enemy}")
            else:
                ens = [(s, x, y) for s, oid, x, y, v in objects(nes)
                       if oid in R.ENEMY_IDS and 0 <= x < 256 and 0 <= y < 240]
                if ens:
                    frozen_enemy = ens[0]
                if death_at is not None and death_at < 0:
                    death_at = None      # survived the previous death
            continue

        n = i - death_at
        if n in (30, 60, 100, 140, 180, 220):
            s, ex, ey = frozen_enemy
            cur = [o for o in objects(nes) if o[0] == s]
            still = snail_at(frame, ex, ey)
            Image.fromarray(frame).resize((768, 720),
                                          Image.NEAREST).save(
                os.path.join(out, f"after_{n:03d}.png"))
            print(f"+{n:3d}: slot{s} now={cur} | snail pixels at "
                  f"({ex:.0f},{ey:.0f})? {still} | D={nes[R.DEATH]}")
        if n > 240:
            break

    nes.controller = 0
    del nes
    print(f"frames in {out}/")


if __name__ == "__main__":
    main()
