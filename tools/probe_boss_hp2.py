"""Find the boss HP bytes in the 1-4 arena.

The arena is unmistakable on screen (14-block HP bar at the top, camera
locked at 247, policy stalls at ~3592 px).  This probe:

  1. drives the policy in god mode ($0072 cleared every NES frame) until
     the camera stops moving  ->  we are in the arena,
  2. logs ALL 2 KB of RAM every decision while spamming right+throw,
  3. dumps the log for offline diffing: the HP bar must be bytes that step
     down each time an axe connects.

    python tools/probe_boss_hp2.py 1-4
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (ACTIONS, AdventureIslandEnv, N_ACTIONS,   # noqa: E402
                 OBS_DIM)
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "models", "ppo_1-3_clear_16M.pt")
OID, ST = R.OBJ.id, R.OBJ.STRIDE
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]


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

    # ---- phase 1: policy in god mode until the camera locks ----------
    action = 0
    frame = 0
    best = 0.0
    stalls = 0
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
    print(f"arena reached: progress={best:.0f} px  camera={int(nes[R.CAMERA])}"
          f"  frame={frame}", flush=True)

    # ---- snapshot the object slots in the arena -----------------------
    print("slots at arena entry:")
    for s in range(16):
        oid = int(nes[OID + ST * s])
        if oid:
            print(f"  slot {s:2d}  id={oid:3d}  x={R.px(R.read_obj_world_x(nes, s)):7.1f}"
                  f"  y={R.px(R.read_obj_world_y(nes, s)):7.1f}")

    # ---- phase 2: spam throws and log full RAM ------------------------
    snap_path = os.path.join(_ROOT, "outputs", "boss_ram_log.npy")
    log = np.zeros((500, 2048), dtype=np.uint8)      # 500 decisions
    n_logged = 0
    for t in range(500):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        act = A_RT if t % 7 in (0, 1) else (A_RJ if t % 23 in (0, 1) else A_RIGHT)
        env.step(act)
        if t % 2 == 0 and n_logged < 500:
            ram = np.array([nes[a] for a in range(2048)], dtype=np.uint8)
            log[n_logged] = ram
            n_logged += 1
    np.save(snap_path, log[:n_logged])
    print(f"logged {n_logged} RAM snapshots -> {snap_path}", flush=True)

    # quick top-level analysis: bytes that changed at all
    changed = [a for a in range(2048) if len(np.unique(log[:n_logged, a])) > 1]
    print(f"bytes that changed at all: {len(changed)}")
    # bytes whose value DECREASED at least once (HP candidates)
    dec = [a for a in changed
           if (np.diff(log[:n_logged, a].astype(int)) < 0).any()]
    print(f"bytes that decreased at least once: {len(dec)}")
    for a in dec:
        vals = log[:n_logged, a]
        steps = np.diff(vals.astype(int))
        downs = int((steps < 0).sum())
        ups = int((steps > 0).sum())
        if downs <= 16 and downs > 0 and ups <= 2:
            uniq = np.unique(vals)
            print(f"  ${a:04X}: min={vals.min()} max={vals.max()} "
                  f"downs={downs} ups={ups} uniq={uniq[:18]}")
    env.close()


if __name__ == "__main__":
    main()
