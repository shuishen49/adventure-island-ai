"""Find the boss HP byte with an ACTIVE headless experiment (v3).

Why this exists
---------------
outputs/manual_ram2.npy was useless: the player died 20 times, ~6 s each,
and never landed a hit.  So we stop waiting for a human and drive the
emulator ourselves:

  * start from saves/stages/stage_1-4_bossface.state  (head id 198 at +99 px,
    body id 67 -- both measured, slots 14 / 15)
  * freeze STAMINA ($0076 = 10) every frame so hunger never kills us.
    No object injection (that perturbs the object table and, as measured on
    2026-10-18, breaks the level-end logic).
  * throw axes rhythmically and detect a hit GEOMETRICALLY: a fast
    right-moving object whose screen box overlaps the head box
  * dump all 2 KB of NES RAM every frame

A byte is an HP candidate when it goes DOWN on >= 60 % of the hit frames,
never goes UP during the fight, and is completely constant in a control run
that never throws.

    python tools/probe_boss_hp3.py --frames 900 --period 24
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import ROM, NES_INPUT_B, load_state_file   # noqa: E402
from cynes import NES                                # noqa: E402

N_RAM = 0x800
HEAD_SLOT = 14          # id 198
BODY_SLOT = 15          # id 67
STAMINA = 0x0076
PROJ_IDS = (66, 68, 69)


def rd(nes, blk: int, slot: int) -> int:
    return int(nes[0x0564 + 0x10 * blk + slot])


def run(nes, state: str, frames: int, period: int, throw: bool):
    nes.load(load_state_file(state))
    dump = np.zeros((frames, N_RAM), dtype=np.uint8)
    hits: list[int] = []
    log: list[str] = []
    prev: dict[int, tuple[int, int]] = {}
    for t in range(frames):
        nes[STAMINA] = 10
        nes[0x0072] = 0                      # clear death flag, stay alive
        nes.controller = (NES_INPUT_B
                          if (throw and t % period == 0) else 0)
        nes.step()
        nes.controller = 0
        for a in range(N_RAM):
            dump[t, a] = nes[a]

        hx, hy = rd(nes, 2, HEAD_SLOT), rd(nes, 6, HEAD_SLOT)
        hid = rd(nes, 0, HEAD_SLOT)
        for s in range(1, 14):
            oid = rd(nes, 0, s)
            if oid not in PROJ_IDS:
                continue
            sx, sy = rd(nes, 2, s), rd(nes, 6, s)
            p = prev.get(s)
            moving = p is not None and sx != p[0]
            if moving and abs(sx - hx) <= 16 and abs(sy - hy) <= 30:
                hits.append(t)
                log.append(f"t={t:4d} hit? slot{s} id{oid} axe=({sx},{sy}) "
                           f"head=({hx},{hy}) head_id={hid}")
            prev[s] = (sx, sy)
    return dump, sorted(set(hits)), log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--period", type=int, default=24)
    ap.add_argument("--state",
                    default=os.path.join(_ROOT, "saves", "stages",
                                         "stage_1-4_bossface.state"))
    args = ap.parse_args()

    nes = NES(ROM)
    ctrl, ctrl_hits, _ = run(nes, args.state, args.frames, args.period, False)
    fight, hits, log = run(nes, args.state, args.frames, args.period, True)

    print(f"control hits={len(ctrl_hits)}  fight hits={len(hits)}  "
          f"frames={args.frames} period={args.period}")
    for line in log[:30]:
        print("  ", line)

    if not hits:
        diff = (fight != ctrl).sum(axis=0)
        top = np.argsort(-diff)[:20]
        print("\nno geometric hit. most-changed addresses fight vs control:")
        for c in top:
            if diff[c] == 0:
                break
            print(f"  0x{c:04X} differs {diff[c]} frames")
        return

    print("\n=== HP candidates ===")
    found = False
    for c in range(N_RAM):
        v = fight[:, c].astype(np.int64)
        u = ctrl[:, c].astype(np.int64)
        if u.min() != u.max():
            continue
        d = np.diff(v)
        mask = np.zeros(len(d), dtype=bool)
        for t in hits:
            if 0 <= t - 1 < len(mask):
                mask[t - 1] = True
        down = int(((d < 0) & mask).sum())
        ups = int((d > 0).sum())
        if down >= max(2, int(0.6 * len(hits))) and ups == 0:
            print(f"  0x{c:04X} down_on_hit={down}/{len(hits)} ups={ups} "
                  f"value@hits={[int(v[t]) for t in hits[:12]]}")
            found = True
    if not found:
        print("  (none)")

    np.save(os.path.join(_ROOT, "outputs", "boss_hp3_fight.npy"), fight)
    np.save(os.path.join(_ROOT, "outputs", "boss_hp3_ctrl.npy"), ctrl)


if __name__ == "__main__":
    main()
