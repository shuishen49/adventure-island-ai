"""Watch a trained PPO agent play Adventure Island in a real window.

    python enjoy.py runs/ppo/ppo_final.pt
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from cynes.windowed import WindowedNES

from env import ACTIONS, OBS_DIM, IMG_H, IMG_W, IMG_STACK, to_grey
import ram_map as R


_CAMS: dict[int, tuple] = {}


def _cam_for(nes):
    """One persistent CameraTracker per emulator instance.

    The NES object is a C extension type and refuses attribute assignment,
    so keep the trackers here.  We also keep a reference to the emulator
    itself, which pins its id and prevents a recycled id from handing a
    stale tracker to a new instance.
    """
    key = id(nes)
    entry = _CAMS.get(key)
    if entry is None or entry[0] is not nes:
        entry = (nes, R.CameraTracker(nes))
        _CAMS[key] = entry
    return entry[1]


def reset_cam(nes) -> None:
    """Drop the cached tracker (call right after (re)booting the game)."""
    _CAMS.pop(id(nes), None)
    _IMGS.pop(id(nes), None)


_IMGS: dict[int, tuple] = {}


def _img_for(nes):
    """Persistent frame stack per emulator (same trick as the camera)."""
    key = id(nes)
    entry = _IMGS.get(key)
    if entry is None or entry[0] is not nes:
        entry = (nes, np.zeros((IMG_STACK, IMG_H, IMG_W), dtype=np.uint8))
        _IMGS[key] = entry
    return entry[1]


def push_img(nes, frame) -> None:
    """Feed a rendered frame into the stack (call it every game frame)."""
    img = _img_for(nes)
    img[:-1] = img[1:]
    img[-1] = to_grey(frame)


def build_obs(nes) -> dict[str, np.ndarray]:
    """Same observation as the training env: {"ram": ..., "img": ...}."""
    import env as E  # reuse internals without owning an emulator

    class _Shim:
        """Minimal duck-type stand-in exposing only what we need."""

        def __init__(self, nes):
            self._nes = nes
            self._start_lives = nes[R.LIVES]
            # A fresh CameraTracker every call would report progress ~0 on
            # every frame -- a train/test mismatch that cripples the policy.
            self._cam = _cam_for(nes)

        _active_enemies = E.AdventureIslandEnv._active_enemies
        _nearest_of_kind = E.AdventureIslandEnv._nearest_of_kind
        _read_score = E.AdventureIslandEnv._read_score
        _track_slots = E.AdventureIslandEnv._track_slots
        _is_zombie = E.AdventureIslandEnv._is_zombie

    shim = _Shim(nes)
    return {"ram": E.AdventureIslandEnv._build_obs(shim),
            "img": _img_for(nes).copy()}


def obs_tensor(nes) -> dict[str, "torch.Tensor"]:
    import torch
    o = build_obs(nes)
    return {"ram": torch.as_tensor(o["ram"], dtype=torch.float32).unsqueeze(0),
            "img": torch.as_tensor(o["img"], dtype=torch.uint8).unsqueeze(0)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_final.pt")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--rom", default=None,
                   help="defaults to env.ROM (the USA ROM)")
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    h = 128
    from train import ActorCritic
    net = ActorCritic(OBS_DIM, len(ACTIONS), h)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"loaded {args.checkpoint} (trained {ck['steps']} env steps)")

    import env as _envmod
    nes = WindowedNES(args.rom or _envmod.ROM, scaling_factor=3)
    # get into the game
    nes.step(frames=300)
    nes.controller = 0x10
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=60)

    reset_cam(nes)
    frame = 0
    action = 0
    while not nes.should_close:
        if frame % 4 == 0:
            with torch.no_grad():
                logits, _ = net(obs_tensor(nes))
                if args.deterministic:
                    action = int(logits.argmax())
                else:
                    action = int(torch.distributions.Categorical(
                        logits=logits).sample())
        nes.controller = ACTIONS[action][0]
        push_img(nes, np.asarray(nes.step(), dtype=np.uint8))
        frame += 1


if __name__ == "__main__":
    sys.exit(main())
