"""What does a fall into a pit look like in the numbers?

Plays with the trained policy until the first death and prints the last
120 frames: player y, gravity, the death register $0072 and the progress.
We need to know whether y grows big enough (and $0072 fires early enough)
to pay a "you are falling" penalty before the episode ends.

Usage:  python tools/probe_fall.py [checkpoint]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cynes import NES  # noqa: E402
import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, ROM  # noqa: E402
from enjoy import build_obs, push_img, reset_cam  # noqa: E402
from train import ActorCritic  # noqa: E402
from visualize import boot  # noqa: E402


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo/ppo_final.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    nes.reset() if hasattr(nes, "reset") else None
    boot(nes)
    reset_cam(nes)
    cam = R.CameraTracker(nes)
    hist = []
    for i in range(4500):
        if i % 4 == 0:
            obs = build_obs(nes)
            with torch.no_grad():
                t = {"ram": torch.as_tensor(obs["ram"], dtype=torch.float32)
                     .unsqueeze(0),
                     "img": torch.as_tensor(obs["img"], dtype=torch.uint8)
                     .unsqueeze(0)}
                action = int(net(t)[0].argmax())
        nes.controller = ACTIONS[action][0]
        push_img(nes, np.asarray(nes.step(), dtype=np.uint8))
        p_y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        grav = nes[R.OBJ.gravity]
        grav = grav - 256 if grav > 127 else grav
        near = []
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        for s in range(1, 16):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid == 0:
                continue
            dx = R.px(R.read_obj_world_x(nes, s) - p_x)
            dy = R.px(R.read_obj_world_y(nes, s) - R.read_obj_world_y(
                nes, R.PLAYER_SLOT))
            if abs(dx) < 48 and abs(dy) < 48:
                near.append(f"{oid}@{dx:+.0f},{dy:+.0f}")
        hist.append((i, cam.progress(nes), p_y, grav, nes[R.DEATH],
                     nes[R.LIVES], ACTIONS[action][1], " ".join(near)))
        if nes[R.DEATH] != 0:
            for _ in range(40):          # keep recording into the fall
                nes.controller = 0
                push_img(nes, np.asarray(nes.step(), dtype=np.uint8))
                p_y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
                hist.append((i, cam.progress(nes), p_y,
                             nes[R.OBJ.gravity], nes[R.DEATH],
                             nes[R.LIVES], "-", ""))
            break
    nes.controller = 0
    del nes

    print(" frame  progress      y  grav death lives act        near(48px)")
    for h in hist[-70:]:
        print(f"{h[0]:6d} {h[1]:9.0f} {h[2]:7.1f} {h[3]:5d} {h[4]:5d} "
              f"{h[5]:5d}  {h[6]:<11} {h[7]}")
    ys = [h[2] for h in hist]
    print(f"\ny range over the whole run: {min(ys):.1f} .. {max(ys):.1f}")


if __name__ == "__main__":
    main()
