"""Push 1-4 to its actual end, unsticking the policy when it wedges.

Why this exists: with real invincibility (fairy / object 199) the 14c
policy scrolls 1-4 out past 25000 px and $007A never fires, while 1-3
clears at ~6600 px.  Two readings:

  * 1-4 really is a 27000 px level   -> unbelievable next to 1-3
  * the level LOOPS and the boss only appears at the true end

The blocker so far is not death (we are invincible) but the policy
WEDGING somewhere around 22000 px.  So: drive it with the policy, and
whenever it stops making progress, inject a burst of random actions to
shake it loose.  Log how far it can get and whether the stage ever ends.

    python tools/probe_14_push.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import AdventureIslandEnv, ACTIONS, N_ACTIONS, OBS_DIM  # noqa
from train import ActorCritic                              # noqa: E402


def force_fairy(nes) -> None:
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 199:
            return
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 0:
            nes[R.OBJ.id + off] = 199
            for f in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr,
                      R.OBJ.y_lo, R.OBJ.y_fr):
                nes[f + off] = nes[f]
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ppo_14c/ppo_final.pt")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    rng = np.random.default_rng(args.seed)

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=args.seed)
    nes = env._nes
    cam = env._cam

    best, stall, unsticks = 0.0, 0, 0
    burst = 0
    for step in range(args.steps):
        force_fairy(nes)
        nes[R.STAMINA] = 10
        nes[R.LIVES] = 9
        nes[R.WEAPON] = max(1, int(nes[R.WEAPON]))
        if burst > 0:
            a = int(rng.integers(0, N_ACTIONS))
            burst -= 1
        else:
            o = env._obs()
            with torch.no_grad():
                a = int(net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)}
                            )[0].argmax())
        nes.controller = ACTIONS[a][0]
        for _ in range(env.frame_skip):
            nes.step()
        nes.controller = 0
        env._push_frame(np.zeros((240, 256, 3), np.uint8))
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
            if stall > 120:
                burst = 45
                unsticks += 1
                stall = 0
        if step % 500 == 0:
            print(f"  step={step:6d} prog={p:8.0f} best={best:8.0f} "
                  f"cam={int(nes[R.CAMERA]):3d} $007A="
                  f"{int(nes[R.STAGE_COMPLETE])} "
                  f"unsticks={unsticks}", flush=True)
        if int(nes[R.STAGE_COMPLETE]) != 0 and int(nes[R.DEATH]) == 0:
            print(f"  >> STAGE COMPLETE at {p:.0f} px (step {step})!",
                  flush=True)
            break
    print(f"done: best={best:.0f} px, unsticks={unsticks}, "
          f"$007A={int(nes[R.STAGE_COMPLETE])}")


if __name__ == "__main__":
    main()
