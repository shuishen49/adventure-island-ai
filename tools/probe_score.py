"""Does the game's own score fire when an enemy dies?

The kill reward currently trusts "an enemy slot vanished within 96 px".
The user asked whether the score would be a better signal, so this plays
a chase-and-shoot policy and prints, for every enemy that disappears:
the score bytes before/after and whether the slot really vanished.

Usage:  python tools/probe_score.py [frames]
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cynes import NES  # noqa: E402
import ram_map as R  # noqa: E402
from env import ACTIONS, ROM  # noqa: E402
from visualize import boot  # noqa: E402

N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
# $069A .. $06A5 : the score / hi-score block
SCORE_BYTES = list(range(R.SCORE, R.SCORE + 12))


def score_bytes(nes) -> list[int]:
    return [nes[a] for a in SCORE_BYTES]


def objects_by_kind(nes, p_x) -> tuple[dict[int, float], dict[int, float]]:
    """(enemies, fruits) as {slot: dx} for the current frame."""
    en, fr = {}, {}
    for s in range(1, 16):
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        if oid in R.ENEMY_IDS:
            en[s] = R.px(R.read_obj_world_x(nes, s) - p_x)
        elif oid in R.FRUIT_IDS:
            fr[s] = R.px(R.read_obj_world_x(nes, s) - p_x)
    return en, fr


def main() -> None:
    nes = NES(ROM)
    boot(nes)
    nes[R.WEAPON] = 1                        # arm the axe

    p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
    prev_e, prev_f = objects_by_kind(nes, p_x)
    prev_score = score_bytes(nes)
    print("score bytes at boot:", prev_score)
    n_kill = n_fruit = n_score = 0
    kill_scored = fruit_scored = 0
    for f in range(N_FRAMES):
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        cur_e, cur_f = objects_by_kind(nes, p_x)
        # aim at the closest enemy ahead
        ahead = [dx for dx in cur_e.values() if -20 < dx < 140]
        if ahead:
            target = min(ahead)
            action = 7 if target < 60 else 9        # right+throw / r+j+throw
        else:
            action = 4                              # right+jump
        nes.controller = ACTIONS[action][0]
        nes.step()

        cur_score = score_bytes(nes)
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        now_e, now_f = objects_by_kind(nes, p_x)

        # enemy / fruit slots that were close and are gone now
        killed = [s for s, dx in prev_e.items()
                  if s not in now_e and abs(dx) < 96]
        eaten = [s for s, dx in prev_f.items()
                 if s not in now_f and abs(dx) < 96]
        scored = [i for i, (a, b) in enumerate(zip(prev_score, cur_score))
                  if a != b]
        n_kill += len(killed)
        n_fruit += len(eaten)
        if scored:
            n_score += 1
        if killed:
            kill_scored += 1 if scored else 0
        if eaten:
            fruit_scored += 1 if scored else 0
        if killed or eaten or scored:
            tag = ("KILL" if killed else "") + ("FRUIT" if eaten else "") \
                  + ("SCORE" if scored else "")
            print(f"f={f:5d} {tag:<11} {prev_score} -> {cur_score}")
        prev_e, prev_f, prev_score = now_e, now_f, cur_score
    nes.controller = 0
    print(f"\n击杀(槽消失) {n_kill} 次, 其中同一帧加分的: {kill_scored}")
    print(f"吃水果(槽消失) {n_fruit} 次, 其中同一帧加分的: {fruit_scored}")
    print(f"分数变化次数: {n_score}")
    print(f"final score bytes {score_bytes(nes)}")
    del nes


if __name__ == "__main__":
    main()
