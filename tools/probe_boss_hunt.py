"""Brute-force search for the action pattern that DAMAGES the 1-4 boss.

Context (measured 2026-10-18)
-----------------------------
* boss head = id 198, slot 14 ; body = id 67, slot 15 (screen-space x).
* From saves/stages/stage_1-4_bossface.state the head sits at screen y = 93,
  i.e. well ABOVE a ground-level axe.  probe_boss_hp3.py threw 25 axes from
  the ground: the head's state bytes ($0672 / $0682) stayed 0 the whole time
  and the run was byte-identical to a no-throw control  ->  no damage.
  So the head has to be reached with a JUMP throw, or at a particular phase
  of its patrol.

This script therefore searches over simple rhythmic macro-programs
(jump period/phase x throw phase x movement) and reports, for every trial,
which RAM bytes moved that a no-input control never touches.  If any trial
changes the head's state bytes -- or makes id 198 disappear -- we have both
the winning technique and the HP byte.

    python tools/probe_boss_hunt.py --trials 240 --frames 300
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

N_RAM = 0x800
HEAD_SLOT, BODY_SLOT = 14, 15
STAMINA, DEATH, STAGE_COMPLETE = 0x0076, 0x0072, 0x007A
WATCH = {0x0572: "head_id", 0x0573: "body_id",
         0x0672: "head_blk16", 0x0673: "body_blk16",
         0x0682: "head_blk17", 0x0683: "body_blk17",
         0x0692: "head_blk18", 0x0693: "body_blk18",
         0x05D2: "head_y", 0x05D3: "body_y"}


def play(nes, state, frames, jump_p, jump_ph, throw_p, throw_ph, move):
    nes.load(load_state_file(state))
    dump = np.zeros((frames, N_RAM), dtype=np.uint8)
    for t in range(frames):
        nes[STAMINA] = 10
        nes[DEATH] = 0
        btn = 0
        if jump_p and t % jump_p == jump_ph:
            btn |= NES_INPUT_A
        if throw_p and t % throw_p == throw_ph:
            btn |= NES_INPUT_B
        if move == "right":
            btn |= NES_INPUT_RIGHT
        elif move == "left":
            btn |= NES_INPUT_LEFT
        elif move == "alt":
            btn |= NES_INPUT_RIGHT if (t // 40) % 2 == 0 else NES_INPUT_LEFT
        nes.controller = btn
        nes.step()
        nes.controller = 0
        for a in range(N_RAM):
            dump[t, a] = nes[a]
    return dump


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=240)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_bossface.state"))
    args = ap.parse_args()

    nes = NES(ROM)
    ctrl = play(nes, args.state, args.frames, 0, 0, 0, 0, "none")
    quiet = np.array([ctrl[:, a].min() == ctrl[:, a].max()
                      for a in range(N_RAM)])
    print(f"control: {int(quiet.sum())} of {N_RAM} bytes are constant")

    grid = list(itertools.product(
        [0, 30, 45, 60, 90],     # jump period (0 = never)
        [0, 15, 30],             # jump phase
        [20, 30, 45],            # throw period
        [0, 10, 20],             # throw phase
        ["none", "right", "left", "alt"]))
    rng = np.random.default_rng(0)
    if len(grid) > args.trials:
        sel = rng.choice(len(grid), args.trials, replace=False)
        grid = [grid[i] for i in sel]

    best = []
    for i, (jp, jph, tp, tph, mv) in enumerate(grid):
        d = play(nes, args.state, args.frames, jp, jph, tp, tph, mv)
        moved = ~np.array([d[:, a].min() == d[:, a].max()
                           for a in range(N_RAM)])
        novel = moved & quiet              # moved here, frozen in control
        head_gone = bool((d[:, 0x0572] != 198).any())
        watch = {n: sorted(set(d[:, a].tolist()))[:8]
                 for a, n in WATCH.items()
                 if len(set(d[:, a].tolist())) > 1}
        dec = 0
        for a in np.where(novel)[0]:
            v = d[:, a].astype(np.int64)
            dd = np.diff(v)
            if (dd < 0).any() and not (dd > 0).any():
                dec += 1
        score = int(novel.sum()) + 100 * head_gone + 10 * dec
        best.append((score, i, (jp, jph, tp, tph, mv), int(novel.sum()),
                     dec, head_gone, watch))
        if head_gone or dec:
            print(f"  ! trial{i} jp={jp} jph={jph} tp={tp} tph={tph} mv={mv} "
                  f"novel={int(novel.sum())} dec={dec} head_gone={head_gone} "
                  f"watch={watch}", flush=True)

    best.sort(reverse=True)
    print("\n=== top trials ===")
    for score, i, prm, nov, dec, hg, watch in best[:12]:
        print(f"  trial{i} score={score} params(jp,jph,tp,tph,mv)={prm} "
              f"novel={nov} dec={dec} head_gone={hg}")
        if watch:
            print(f"      watch={watch}")


if __name__ == "__main__":
    main()
