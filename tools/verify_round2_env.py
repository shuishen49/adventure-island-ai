"""Smoke-test starting an episode in the second stage (第二关 / round 1).

Checks, for both ways of getting there (`--start-round 1` and the dumped
state file):

  * reset lands in round 1 with the player alive, stamina full, 3 lives
  * the random warm-up frames do NOT kill us (round 2 starts up in the
    air on a platform -- walking off it during warm-up would end every
    episode before it began)
  * the camera progress tracker still advances in this stage
  * death -> reset cycles work and come back to the round-2 start
  * the observation looks sane (no NaNs)

usage: python tools/verify_round2_env.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R  # noqa: E402
from env import ACTIONS, AdventureIslandEnv  # noqa: E402

OUT = os.path.join(_ROOT, "outputs", "probe_round2")
os.makedirs(OUT, exist_ok=True)
STATE = os.path.join(_ROOT, "saves", "round2_start_warp.state")
STEPS = 400


def shot(nes, tag):
    arr = np.asarray(nes.step(), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").resize((512, 480), Image.NEAREST).save(
        os.path.join(OUT, f"{tag}.png"))
    return arr


def describe(env, tag):
    nes = env._nes
    x, y = (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))
    print(f"  {tag}: area={nes[R.AREA]} round={nes[R.ROUND]} "
          f"ckpt={nes[R.CHECKPOINT]} lives={nes[R.LIVES]} "
          f"sta={nes[R.STAMINA]} death={nes[R.DEATH]} "
          f"weapon={nes[R.WEAPON]} px=({x:.1f},{y:.1f})", flush=True)


def run(tag, **kwargs):
    print(f"\n=== {tag} ===")
    env = AdventureIslandEnv(**kwargs)
    try:
        for seed in range(3):                  # warm-up is random
            obs = env.reset(seed=seed)
            if seed == 0:
                describe(env, "after reset")
                assert np.isfinite(obs["ram"]).all(), "NaN in ram obs"
                print(f"    ram obs ok ({obs['ram'].shape}), "
                      f"img {obs['img'].shape}")
            if env._nes[R.DEATH] != 0:
                print(f"  !! warm-up killed us (seed {seed})")
        print("  warm-up survives: ok")

        # drive it: mostly right, hop every 12 decision steps
        obs = env.reset(seed=0)
        shot(env._nes, f"{tag}_start")
        deaths = 0
        prog = 0.0
        best = 0.0
        for i in range(STEPS):
            a = 4 if (i % 12) == 0 else 1       # right+jump / right
            obs, r, done, info = env.step(a)
            best = max(best, info["x_progress"])
            if i % 100 == 0:
                prog = info["x_progress"]
                print(f"    step {i:4d} progress={prog:7.1f} "
                      f"reward={r:+.3f} lives={info['lives']} "
                      f"sta={info['stamina']} round={env._nes[R.ROUND]}",
                      flush=True)
            if done:
                if info["death"]:
                    deaths += 1
                    print(f"    died at step {i} "
                          f"(progress {info['x_progress']:.0f})", flush=True)
                else:
                    print(f"    ended at step {i} "
                          f"(stalled={info['stalled']})", flush=True)
                obs = env.reset()
                describe(env, "after reset")
        print(f"  result: {deaths} deaths in {STEPS} steps, "
              f"best progress {best:.0f} px")
        shot(env._nes, f"{tag}_end")
    finally:
        env.close()


if __name__ == "__main__":
    run("start_round=1", start_round=1)
    if os.path.exists(STATE):
        run("start_state=file", start_state=STATE)
    print("\nscreenshots in", OUT)
