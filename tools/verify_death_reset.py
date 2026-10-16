"""Verify the two new env behaviours end-to-end:

  1. the episode ends on the death FLAG, not 236 frames later on the
     lives counter;
  2. every reset starts with the axe armed ($006D == 1) and the agent can
     actually throw it (id-66 projectile appears when B is tapped).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from cynes import NES_INPUT_B

import ram_map as R
from env import ACTIONS, AdventureIslandEnv

B_IDX = next(i for i, a in enumerate(ACTIONS) if a[1] == "throw")


def main() -> None:
    env = AdventureIslandEnv()
    obs = env.reset()
    nes = env._nes
    print(f"after reset: weapon={nes[R.WEAPON]} "
          f"obs[weapon]={obs['ram'][5]:.1f}")

    # --- 1. can it throw right away? -----------------------------------
    # B is edge-triggered: alternate throw / noop so the button is released
    # between shots (holding it down fires exactly once).
    threw = 0
    for i in range(60):
        _, _, done, _ = env.step(B_IDX if i % 3 == 0 else 0)
        if any(nes[R.OBJ.id + s] == 66 for s in range(16)):
            threw += 1
        if done:
            break
    print(f"axe-in-flight sightings while tapping B: {threw}")

    # --- 2. instant death termination ----------------------------------
    env = AdventureIslandEnv()
    env.reset()
    nes = env._nes
    rng = np.random.default_rng(0)
    death_step = None
    done_step = None
    for s in range(4000):
        _, _, done, info = env.step(int(rng.integers(0, len(ACTIONS))))
        if death_step is None and nes[R.DEATH] != 0:
            death_step = s
        if done:
            done_step = s
            print(f"\nepisode ended at step {done_step}; info={info}")
            break
    if death_step is None:
        print("no death reached")
        return
    lag = (done_step - death_step) if done_step is not None else None
    print(f"death flag at step {death_step}, episode ended at {done_step} "
          f"-> lag {lag} env steps "
          f"({None if lag is None else lag * env.frame_skip} frames)")
    print(f"lives at that moment: {nes[R.LIVES]} (was 3 at boot) "
          "-- unchanged means we did NOT wait for the counter")

    # --- 3. episode count / throughput ---------------------------------
    env = AdventureIslandEnv()
    env.reset()
    rng = np.random.default_rng(1)
    lens = []
    steps = 0
    for s in range(2000):
        _, _, done, _ = env.step(int(rng.integers(0, len(ACTIONS))))
        steps += 1
        if done:
            lens.append(steps)
            steps = 0
            env.reset()
    print(f"\n2000 random steps -> {len(lens)} episodes, "
          f"mean length {np.mean(lens):.0f} steps")


if __name__ == "__main__":
    main()
