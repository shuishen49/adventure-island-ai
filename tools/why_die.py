"""Dump the game state at the exact frame the player dies.

The generic diagnose.py could not classify the deaths, so this logs the
raw numbers: death register $0072, player y, stamina and every object
within 40 px of the player.

Usage:  python tools/why_die.py [checkpoint] [episodes]
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


def snapshot(nes) -> str:
    p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
    p_y = R.read_obj_world_y(nes, R.PLAYER_SLOT)
    near = []
    for s in range(1, 16):
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        if oid == 0:
            continue
        dx = R.px(R.read_obj_world_x(nes, s) - p_x)
        dy = R.px(R.read_obj_world_y(nes, s) - p_y)
        if abs(dx) < 48 and abs(dy) < 48:
            near.append(f"{oid}@{dx:+.0f},{dy:+.0f}")
    return (f"death={nes[R.DEATH]:3d} y={R.px(p_y):7.1f} "
            f"sta={nes[R.STAMINA]:2d} cam={nes[R.CAMERA]:3d} "
            f"near=[{' '.join(near)}]")


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo/ppo_final.pt"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"{ckpt} ({ck['steps']} steps), {n_ep} episodes\n")

    nes = NES(ROM)
    rng = np.random.default_rng(0)
    for e in range(n_ep):
        nes.reset() if hasattr(nes, "reset") else None
        boot(nes)
        reset_cam(nes)
        cam = R.CameraTracker(nes)
        last_lives = nes[R.LIVES]
        print(f"--- ep{e + 1} ---")
        for i in range(4500):
            if i < 120:
                action = int(rng.integers(len(ACTIONS)))
            elif i % 4 == 0:
                obs = build_obs(nes)
                with torch.no_grad():
                    t = {"ram": torch.as_tensor(obs["ram"], dtype=torch.float32)
                         .unsqueeze(0),
                         "img": torch.as_tensor(obs["img"], dtype=torch.uint8)
                         .unsqueeze(0)}
                    action = int(net(t)[0].argmax())
            nes.controller = ACTIONS[action][0]
            push_img(nes, np.asarray(nes.step(), dtype=np.uint8))
            cam.progress(nes)
            lv = nes[R.LIVES]
            if lv < last_lives:
                last_lives = lv
                print(f"  帧{i:5d} 进度{cam.progress(nes):7.0f}  "
                      f"{snapshot(nes)}")
            if lv == 0:
                break
        nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
