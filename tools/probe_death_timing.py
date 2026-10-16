"""When exactly does the game register a death, and what gets lost?

Drives the real env (frame_skip=1) with random actions until it dies,
then prints a frame-by-frame trace around the death: lives ($003F),
death flag ($0072), weapon ($006D), player y.  Answers two questions:

  1. how many frames elapse between the death instant ($0072 != 0) and
     the lives decrement -- that is the window the agent wastes
     watching the death + respawn animation;
  2. is the axe ($006D) gone after the respawn?
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import ram_map as R
from env import ACTIONS, AdventureIslandEnv


def main() -> None:
    env = AdventureIslandEnv(frame_skip=1)
    obs = env.reset()
    nes = env._nes
    base_lives = nes[R.LIVES]
    print(f"boot: lives={base_lives} $72={nes[R.DEATH]} "
          f"weapon={nes[R.WEAPON]} y_lo={nes[R.OBJ.y_lo]}")

    rng = np.random.default_rng(0)
    trace = []
    death_frame = None
    for f in range(4000):
        a = int(rng.integers(0, len(ACTIONS)))
        _, _, done, _ = env.step(a)
        trace.append((f, nes[R.LIVES], nes[R.DEATH], nes[R.WEAPON],
                      nes[R.OBJ.y_lo], nes[R.STAMINA]))
        if death_frame is None and nes[R.DEATH] != 0:
            death_frame = f
        if done:
            print(f"env said DONE at step {f}")
            break
        if death_frame is not None and f > death_frame + 300:
            break

    if death_frame is None:
        print("no death observed")
        return

    print(f"\ndeath flag ($0072) first non-zero at step {death_frame}")
    lo = max(0, death_frame - 3)
    hi = min(len(trace), death_frame + 120)
    print(" step  lives  $72  weapon  y_lo  stamina")
    for f, lv, dth, w, y, st in trace[lo:hi]:
        mark = "   <-- $72 set" if f == death_frame else ""
        print(f"{f:5d} {lv:6d} {dth:4d} {w:7d} {y:5d} {st:8d}{mark}")

    lives_step = next((f for f, lv, *_ in trace if lv < base_lives), None)
    print(f"\nlives {base_lives} -> {nes[R.LIVES]} at step {lives_step} "
          f"(lag = {None if lives_step is None else lives_step - death_frame} "
          f"frames after the death flag)")
    print(f"weapon just before death: {trace[max(0, death_frame - 1)][3]}"
          f", now: {nes[R.WEAPON]}")


if __name__ == "__main__":
    main()
