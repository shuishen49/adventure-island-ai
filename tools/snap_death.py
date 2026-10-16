"""Save the annotated frame of the moment the agent dies.

Shows what actually kills it (2026-10-13: a snail 13 px away, on the
ground -- not a pit).

Usage:  python tools/snap_death.py [checkpoint]
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
from visualize import annotate, boot  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo/ppo_final.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # an old 8-action checkpoint loads into an 8-head network: the first 8
    # entries of ACTIONS are unchanged, so the action indices still match
    n_actions = ck["model"]["policy.weight"].shape[0]
    net = ActorCritic(OBS_DIM, n_actions, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    nes.reset() if hasattr(nes, "reset") else None
    boot(nes)
    reset_cam(nes)
    cam = R.CameraTracker(nes)
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
        buf = np.asarray(nes.step(), dtype=np.uint8)
        push_img(nes, buf)
        cam.progress(nes)
        if nes[R.DEATH] != 0:
            img = annotate(buf, nes, ACTIONS[action][1])
            from PIL import Image
            Image.fromarray(img).resize((256 * 3, 240 * 3),
                                        Image.NEAREST).save(
                os.path.join(OUT, "death_snail.png"))
            print(f"saved outputs/death_snail.png "
                  f"(frame {i}, progress {cam.progress(nes):.0f} px)")
            break
    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
