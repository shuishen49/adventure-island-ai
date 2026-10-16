"""Does $006D really mean "I am holding the axe"?

Play with a right-running / jumping / throwing policy and log every frame
where the WEAPON byte changes, together with what was nearby just before
the change (so we can tell whether an id-28 item was picked up) and what
happens when the throw button is used.

Usage:  python tools/probe_weapon.py [frames]
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

from cynes import (
    NES,
    NES_INPUT_A,
    NES_INPUT_B,
    NES_INPUT_RIGHT,
    NES_INPUT_START,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ram_map as R  # noqa: E402

ROM = os.environ.get(
    "AI_ROM",
    r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes",
)
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000


def nearby(nes):
    """[(id, dx_px, dy_px)] of live object slots close to the player."""
    p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
    p_y = R.read_obj_world_y(nes, R.PLAYER_SLOT)
    out = []
    for s in range(1, 16):
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        if oid == 0:
            continue
        dx = R.px(R.read_obj_world_x(nes, s) - p_x)
        dy = R.px(R.read_obj_world_y(nes, s) - p_y)
        if abs(dx) < 40 and abs(dy) < 40:
            out.append((oid, round(dx, 1), round(dy, 1)))
    return out


def main() -> None:
    nes = NES(ROM)
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=60)

    rng = random.Random(7)
    prev_w = nes[R.WEAPON]
    prev_sk = nes[R.SKATEBOARD]
    prev_lives = nes[R.LIVES]
    hist: list[list] = []          # ring buffer of last frames' nearby sets
    events: list[tuple] = []
    throw_after: dict[int, int] = {}
    w_hist: dict[int, int] = {}

    for f in range(N_FRAMES):
        # mostly run right, jump now and then, throw when armed
        ctl = NES_INPUT_RIGHT
        if rng.random() < 0.12:
            ctl |= NES_INPUT_A
        if rng.random() < 0.06:
            ctl |= NES_INPUT_B
        nes.controller = ctl
        nes.step()

        hist.append(nearby(nes))
        if len(hist) > 12:
            hist.pop(0)

        w = nes[R.WEAPON]
        w_hist[w] = w_hist.get(w, 0) + 1
        if w != prev_w:
            events.append((
                f, "WEAPON", prev_w, w,
                nes[R.LIVES], nes[R.STAMINA], nes[R.SKATEBOARD],
                hist[max(0, len(hist) - 4)],
                hist[-1],
            ))
            prev_w = w
        sk = nes[R.SKATEBOARD]
        if sk != prev_sk:
            events.append((
                f, "SKATE", prev_sk, sk, nes[R.LIVES], nes[R.STAMINA], w,
                hist[max(0, len(hist) - 4)], hist[-1],
            ))
            prev_sk = sk
        if nes[R.LIVES] != prev_lives:
            events.append((
                f, "LIVES", prev_lives, nes[R.LIVES], nes[R.LIVES],
                nes[R.STAMINA], w, [], hist[-1],
            ))
            prev_lives = nes[R.LIVES]
            prev_w = nes[R.WEAPON]      # respawn resets everything

    print(f"frames={N_FRAMES}  WEAPON value histogram: {sorted(w_hist.items())}")
    print(f"events: {len(events)}")
    for e in events[:60]:
        f, kind, a, b, lives, sta, extra, before, after = e
        print(f"  f={f:>7} {kind:6} {a} -> {b}  lives={lives} sta={sta} "
              f"extra={extra}")
        print(f"            before={before}")
        print(f"            after ={after}")


if __name__ == "__main__":
    main()
