"""Decisive test: is $0011 really the boss HP?

probe_boss_v3.py showed $0011 stepping down 8 -> 1 exactly while jump-throws
were landing, then freezing the moment the player died (id 198 appeared).
If $0011 is the HP counter, poking it to 0 must end the fight.

    python tools/probe_boss_confirm.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (ACTIONS, AdventureIslandEnv, N_ACTIONS,   # noqa: E402
                 OBS_DIM)
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "models", "ppo_1-3_clear_16M.pt")
OID, ST = R.OBJ.id, R.OBJ.STRIDE


def shot(nes, name: str) -> None:
    arr = np.asarray(nes.step(), dtype=np.uint8).copy()
    Image.fromarray(arr).convert("RGB").resize((1024, 960), Image.NEAREST)\
        .save(os.path.join(_ROOT, "outputs", name))


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_round=3, start_with_axe=True,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    o = env._obs()
    action, frame, best, stalls = 0, 0, 0.0, 0
    while frame < 60 * 60 * 4:
        if frame % env.frame_skip == 0:
            with torch.no_grad():
                lg, _ = net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)})
            action = int(lg[0].argmax())
        nes.controller = ACTIONS[action][0]
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        arr = np.asarray(nes.step(), dtype=np.uint8)
        nes.controller = 0
        if frame % env.frame_skip == env.frame_skip - 1:
            env._push_frame(arr)
        o = env._obs()
        frame += 1
        p = env._cam.progress(nes)
        if p > best + 0.5:
            best, stalls = p, 0
        else:
            stalls += 1
        if stalls > 45 * 60:
            break
    print(f"arena: progress={best:.0f}  $0011={int(nes[0x11])}", flush=True)
    shot(nes, "boss_confirm_before.png")

    # --- the poke ------------------------------------------------------
    nes[0x11] = 0
    print("poke $0011 = 0")
    cleared = False
    for t in range(400):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        env.step(0)                      # stand still
        if t % 40 == 0:
            print(f"  t={t:3d} $0011={int(nes[0x11]):3d} "
                  f"STAGE_COMPLETE={int(nes[R.STAGE_COMPLETE])} "
                  f"lives={int(nes[R.LIVES])} score={int(nes[R.SCORE])}",
                  flush=True)
        if int(nes[R.STAGE_COMPLETE]) != 0:
            cleared = True
            print(f"  *** STAGE COMPLETE after {t} decisions ***", flush=True)
            break
    shot(nes, "boss_confirm_after.png")
    print("boss cleared:", cleared)
    env.close()


if __name__ == "__main__":
    main()
