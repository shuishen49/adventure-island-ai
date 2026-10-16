"""Prove that the "+1.0 per kill" reward really fires on a real kill.

Drives the env with a chase-and-shoot controller and saves an annotated
frame at the exact step where info["kills"] > 0, together with the reward
that was paid.  If the picture shows our axe (SHOT 66) on top of where a
snail (ENEMY 14) just was, the detector is sound.

Usage:  python tools/snap_kill.py [steps]
"""

from __future__ import annotations

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import ACTIONS, AdventureIslandEnv  # noqa: E402
from visualize import annotate  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


def main() -> None:
    env = AdventureIslandEnv(seed=3)
    obs = env.reset(seed=3)
    shots = 0
    for i in range(N):
        e_dx, e_dy, e_seen = obs["ram"][-6:-3]
        if e_seen and 0 < e_dx < 140:
            action = 9 if e_dx > 45 else 7     # right+jump+throw / right+throw
        else:
            action = 4                         # right+jump
        obs, reward, done, info = env.step(action)
        if info["kills"]:
            shots += 1
            img = annotate(env._last_frame, env._nes, ACTIONS[action][1])
            Image.fromarray(img).resize((256 * 3, 240 * 3),
                                        Image.NEAREST).save(
                os.path.join(OUT, f"kill_{shots}.png"))
            print(f"step {i:4d}: kills={info['kills']}  reward={reward:+.2f}  "
                  f"progress={info['x_progress']:.0f} px  "
                  f"-> outputs/kill_{shots}.png")
            if shots >= 2:
                break
        if done:
            print(f"episode over at step {i}, restarting")
            obs = env.reset()
    if not shots:
        print("no kill was detected in", N, "steps")
    env.close()


if __name__ == "__main__":
    main()
