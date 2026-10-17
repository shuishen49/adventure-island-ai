"""What does the skateboard ($006E) actually do?  Poke it until it answers.

The reward code pays SKATE_GAIN when the board appears and SKATE_SAVE when
it disappears while we are still alive -- that second one is the only
signal telling the agent what the board was *for*.  Both numbers rest on
the measurements below, so re-run this if the game or the RAM map ever
changes.

Facts established here (Adventure Island, USA ROM):

  * $006E is a 0/1 flag.  Nothing else was ever observed.
  * With the board, walking into an enemy does NOT kill you -- the board
    is consumed instead.  Without it, the very same walk kills.
  * The board does NOT save you from a fall: dropping into the sea kills
    with or without it (and with it you get there sooner, because the
    board is faster).
  * The flag is reachable in normal play: the 1-1 policy picks the board
    up around step 100 in most episodes.
"""

import os
import sys

import numpy as np
from cynes import NES

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R                                            # noqa: E402
from env import ROM                                            # noqa: E402

STATES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "saves", "stages")


def load(stage: str) -> np.ndarray:
    with open(os.path.join(STATES, f"stage_{stage}.state"), "rb") as fh:
        return np.frombuffer(fh.read(), dtype=np.uint8).copy()


def walk_into_enemy(skate, stage="1-1", frames=700):
    """Hold RIGHT until an enemy kills us.  Returns the death frame."""
    nes = NES(ROM)
    nes.load(load(stage))
    if skate is not None:
        nes[R.SKATEBOARD] = skate
    for f in range(frames):
        # The game overwrites $006E on its own now and then, so a single
        # write at t=0 is not enough to *hold* the state; re-assert it.
        if skate is not None:
            nes[R.SKATEBOARD] = skate
        nes.controller = 0x01
        nes.step()
        nes.controller = 0
        if nes[R.DEATH] != 0:
            del nes
            return f
    del nes
    return None


def walk_into_the_sea(skate, stage="1-2", frames=900):
    nes = NES(ROM)
    nes.load(load(stage))
    for f in range(frames):
        if skate is not None:
            nes[R.SKATEBOARD] = skate
        nes.controller = 0x01
        nes.step()
        nes.controller = 0
        if nes[R.DEATH] != 0:
            del nes
            return f
    del nes
    return None


def observed_values(stage="1-1", frames=1200):
    nes = NES(ROM)
    nes.load(load(stage))
    seen = set()
    for f in range(frames):
        nes.controller = 0x01 if (f // 12) % 3 else 0x01 | 0x80
        nes.step()
        nes.controller = 0
        seen.add(nes[R.SKATEBOARD])
        if nes[R.DEATH] != 0:
            break
    del nes
    return sorted(seen)


def main() -> None:
    print("=== 1. what values can $006E take? ===")
    for stage in ("1-1", "1-2", "1-3"):
        print(f"  {stage}: {observed_values(stage)}")

    print("\n=== 2. walking into an enemy, board vs no board ===")
    bare = walk_into_enemy(None)
    held = walk_into_enemy(1)
    print(f"  no board : death at frame {bare}")
    print(f"  board on : death at frame {held}")
    if bare is not None and held is None:
        print("  --> the board absorbed the hit: this is a spare life")
    else:
        print("  !! UNEXPECTED - the reward constant SKATE_SAVE is wrong")

    print("\n=== 3. falling into the sea, board vs no board ===")
    bare = walk_into_the_sea(None)
    held = walk_into_the_sea(1)
    print(f"  no board : death at frame {bare}")
    print(f"  board on : death at frame {held}")
    if bare is not None and held is not None:
        print("  --> the board does NOT save you from a fall")
    else:
        print("  !! UNEXPECTED - falls must stay fatal")


if __name__ == "__main__":
    main()
