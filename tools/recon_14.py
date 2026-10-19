"""Push into 1-4 far enough to see whether there is a boss at all.

Earlier "1-4 has no boss" was measured with a scripted bot that died at
696 px, plus a god-mode run that got stuck in the death animation -- so it
never actually reached the arena.  Hudson's Adventure Island puts a boss at
the end of every 4th round, so 1-4 should have one.

This drives the REAL policy (far better than any scripted bot) and clears
$0072 on EVERY NES frame so it cannot die, then dumps frames every 500 px
and logs the object ids it meets.

    python tools/recon_14.py            # writes outputs/recon14_*.png
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

    best = 0.0
    shots: dict[int, str] = {}
    ids_seen: dict[int, int] = {}
    action = 0
    frame = 0
    stalls = 0
    while frame < 60 * 60 * 6:            # 6 minutes of game time
        if frame % env.frame_skip == 0:
            with torch.no_grad():
                lg, _ = net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)})
            action = int(lg[0].argmax())
        nes.controller = ACTIONS[action][0]
        nes[R.DEATH] = 0                  # god mode, every NES frame
        nes[R.STAMINA] = 10
        arr = np.asarray(nes.step(), dtype=np.uint8)
        nes.controller = 0
        if frame % env.frame_skip == env.frame_skip - 1:
            env._push_frame(arr)
        o = env._obs()
        frame += 1

        p = env._cam.progress(nes)
        if p > best + 0.5:
            best = p
            stalls = 0
        else:
            stalls += 1
        for s in range(1, 16):
            oid = int(nes[OID + ST * s])
            if oid:
                ids_seen[oid] = ids_seen.get(oid, 0) + 1
        mark = int(best // 500) * 500
        if mark and mark not in shots:
            img = Image.fromarray(arr).convert("RGB").resize(
                (1024, 960), Image.NEAREST)
            path = os.path.join(_ROOT, "outputs", f"recon14_{mark:05d}.png")
            img.save(path)
            shots[mark] = path
            print(f"  {mark:5d} px -> {path}", flush=True)
        if int(nes[R.STAGE_COMPLETE]) != 0:
            print(f"STAGE_COMPLETE at {best:.0f} px (frame {frame})")
            break
        if stalls > 60 * 60:              # a minute with no progress
            print(f"stalled at {best:.0f} px (frame {frame}) -- "
                  f"camera={int(nes[R.CAMERA])} death={int(nes[R.DEATH])}")
            img = Image.fromarray(arr).convert("RGB").resize(
                (1024, 960), Image.NEAREST)
            img.save(os.path.join(_ROOT, "outputs", "recon14_stall.png"))
            break
    print(f"best {best:.0f} px")
    print("ids:", sorted(ids_seen.items(), key=lambda kv: -kv[1])[:14])
    env.close()


if __name__ == "__main__":
    main()
