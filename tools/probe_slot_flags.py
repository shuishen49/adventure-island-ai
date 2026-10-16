"""Probe: find the per-slot ACTIVE flag.

We know stale object slots survive a death/respawn with frozen coordinates
and are no longer rendered by the game.  Compare every candidate field byte
of one slot while it is alive vs after respawn, to find the byte the game
flips when a slot becomes inactive.
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from cynes import NES

import ram_map as R
from env import ACTIONS, ROM, OBS_DIM
from enjoy import obs_tensor, reset_cam
from train import ActorCritic
from visualize import boot, objects

FIELD_BASES = [0x0604, 0x0614, 0x0624, 0x0634, 0x0644, 0x0654, 0x0664]


def slot_dump(nes, s):
    oid = nes[R.OBJ.id + s]
    fields = " ".join(f"{b:04X}={nes[b + s]:3d}" for b in FIELD_BASES)
    grav = nes[R.OBJ.gravity + s]
    return (f"s{s} id={oid:3d} grav={grav:3d} " + fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_00092160.pt")
    ap.add_argument("--max-frames", type=int, default=12000)
    ap.add_argument("--wait-after", type=int, default=350)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    boot(nes)
    reset_cam(nes)

    action = 0
    death_frame = None
    watch = None          # slot we follow (an onscreen enemy at death time)
    prev_lives = nes[R.LIVES]

    for i in range(args.max_frames):
        if i % 4 == 0:
            with torch.no_grad():
                action = int(net(obs_tensor(nes))[0].argmax())
        nes.controller = ACTIONS[action][0]
        nes.step()

        lv = nes[R.LIVES]
        objs = objects(nes)
        enemies = [o for o in objs if o[1] in R.ENEMY_IDS and o[0] != 0]

        if death_frame is None:
            # before death: follow the first onscreen enemy, log it every 30f
            if enemies and watch is None:
                watch = enemies[0][0]
            if watch is not None and i % 30 == 0:
                print(f"f{i:5d} lives={lv} ALIVE  {slot_dump(nes, watch)}")
            if lv < prev_lives:
                print(f"f{i:5d} === DEATH (lives {lv}) ===")
                if watch is None:
                    print("no onscreen enemy to follow, abort")
                    return
                death_frame = i
        else:
            n = i - death_frame
            if n % 25 == 0:
                print(f"f{i:5d} (+{n:3d}) lives={lv} D={nes[R.DEATH]:3d} "
                      f"{slot_dump(nes, watch)}")
            if n >= args.wait_after:
                break
        prev_lives = lv

    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
