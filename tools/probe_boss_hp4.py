"""Confirm $0011 is the 1-4 boss HP -- by actually LANDING hits.

Evidence it is HP (2026-10-18):
  saves/stages/stage_1-4arena.state   (made with --no-throw)  $0011 = 8
  saves/stages/stage_1-4arena_b.state  (made while throwing)   $0011 = 7
  saves/stages/stage_1-4_bossface.state                  $0011 = 6
  and the game itself needs 8 head hits for the area-1 boss
  ( Evil Witch Doctor; 8 + 2 per area ).

The earlier "it is a countdown, not HP" verdict was wrong: it only
decrements when the boss is actually damaged (600 axes thrown in
probe_boss_hp3.py left it pinned at 6).

So: search for a behaviour that decrements it.  Modes
  grid  - rhythmic jump/throw programs
  chase - walk toward the head until it is close, then jump+throw

    python tools/probe_boss_hp4.py --mode chase --trials 60
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import (ROM, NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT,  # noqa: E402
                 NES_INPUT_RIGHT, load_state_file)
from cynes import NES                                            # noqa: E402

HP = 0x0011
STAMINA, DEATH = 0x0076, 0x0072
HEAD_SLOT = 14
X_LO, Y_LO = 0x0584, 0x05C4


def head(nes):
    return (int(nes[X_LO + HEAD_SLOT]), int(nes[Y_LO + HEAD_SLOT]))


def player_x(nes):
    return int(nes[X_LO])


def play(nes, state, frames, mode, jp, jph, tp, tph, mv, close_to):
    nes.load(load_state_file(state))
    hp0 = int(nes[HP])
    traj = [hp0]
    drops = []
    for t in range(frames):
        nes[STAMINA] = 10
        nes[DEATH] = 0
        btn = 0
        hx, hy = head(nes)
        px = player_x(nes)
        dx = hx - px
        if mode == "chase":
            btn |= NES_INPUT_RIGHT if dx > close_to else 0
            if dx <= close_to + 24:
                if jp and t % jp == jph:
                    btn |= NES_INPUT_A
                if tp and t % tp == tph:
                    btn |= NES_INPUT_B
        else:
            if jp and t % jp == jph:
                btn |= NES_INPUT_A
            if tp and t % tp == tph:
                btn |= NES_INPUT_B
            if mv == "right":
                btn |= NES_INPUT_RIGHT
            elif mv == "left":
                btn |= NES_INPUT_LEFT
            elif mv == "alt":
                btn |= (NES_INPUT_RIGHT if (t // 40) % 2 == 0
                        else NES_INPUT_LEFT)
        nes.controller = btn
        nes.step()
        nes.controller = 0
        v = int(nes[HP])
        if v != traj[-1]:
            drops.append((t, traj[-1], v, dx, hy))
        traj.append(v)
    return hp0, min(traj), drops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_bossface.state"))
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--mode", default="chase")
    ap.add_argument("--close-to", type=int, default=48)
    ap.add_argument("--trials", type=int, default=60)
    args = ap.parse_args()

    nes = NES(ROM)
    combos = list(itertools.product(
        [0, 24, 36, 48, 72], [0, 12, 24], [12, 20, 30, 45], [0, 6, 12],
        ["none", "right"]))
    rng = np.random.default_rng(1)
    if len(combos) > args.trials:
        combos = [combos[i]
                  for i in rng.choice(len(combos), args.trials, False)]

    wins = []
    for i, (jp, jph, tp, tph, mv) in enumerate(combos):
        hp0, hpmin, drops = play(nes, args.state, args.frames, args.mode,
                                 jp, jph, tp, tph, mv, args.close_to)
        if drops:
            wins.append((hp0 - hpmin, i, (jp, jph, tp, tph, mv), drops[:8]))
            print(f"  ! DAMAGE trial{i} jp={jp} jph={jph} tp={tp} tph={tph} "
                  f"mv={mv}: {hp0} -> {hpmin}   {drops[:8]}", flush=True)
    print(f"\n{len(wins)} / {len(combos)} trials damaged the boss")
    wins.sort(reverse=True)
    for dmg, i, prm, drops in wins[:5]:
        print(f"  best: trial{i} {prm} dmg={dmg} drops={drops}")


if __name__ == "__main__":
    main()
