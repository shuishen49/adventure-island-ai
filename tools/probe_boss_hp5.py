"""Land a real hit on the 1-4 boss head, using the RED FLASH as ground truth.

What we know now (2026-10-18, measured)
--------------------------------------
* boss head = id 198 (slot 14), body = id 67 (slot 15); head sits 40 px
  above the body -- from stage_1-4_bossface.state: head y=93, body y=133.
* a STANDING throw releases the axe at y=136 and it only falls  -> never
  reaches the head.
* a standing JUMP throw releases it at y=107 -- still 14 px below the head.
* the manual says Higgins jumps HIGHER while running, so the winning move
  should be: run right -> jump -> throw at the apex.
* a successful hit turns the head RED (documented boss behaviour), which
  gives us a pixel-level ground truth instead of guessing at RAM bytes.

So: grid over (run length, jump moment, throw delay), freeze hunger/death,
capture every frame, and flag trials whose red-pixel count inside the head
box spikes.  For those trials we then hunt for the HP byte: a byte that goes
down on the flash frame and never comes back up.

    python tools/probe_boss_hp5.py --trials 400
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

from env import (ROM, NES_INPUT_A, NES_INPUT_B, NES_INPUT_RIGHT,  # noqa: E402
                 load_state_file)
from cynes import NES                                            # noqa: E402

N_RAM = 0x800
STAMINA, DEATH, HP = 0x0076, 0x0072, 0x0011
ID, XL, YL = 0x0564, 0x0584, 0x05C4
HEAD_SLOT = 14


def red_count(frame: np.ndarray, cx: int, cy: int, rx=20, ry=24) -> int:
    """How many strongly-red pixels sit in the head box (0..255 coords)."""
    h, w = frame.shape[0], frame.shape[1]
    x0, x1 = max(0, cx - rx), min(w, cx + rx)
    y0, y1 = max(0, cy - ry), min(h, cy + ry)
    if x1 <= x0 or y1 <= y0:
        return 0
    p = frame[y0:y1, x0:x1].astype(np.int32)
    if p.ndim == 2:
        return 0
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    return int(((r > 90) & (r > g + 40) & (r > b + 40)).sum())


def play(nes, state, frames, run_len, jump_at, throw_delay, dump=False):
    nes.load(load_state_file(state))
    reds: list[int] = []
    hps: list[int] = []
    head_xy: list[tuple[int, int]] = []
    dump_buf = (np.zeros((frames, N_RAM), dtype=np.uint8) if dump else None)
    hit_geom = []
    for t in range(frames):
        nes[STAMINA] = 10
        nes[DEATH] = 0
        btn = 0
        if t < run_len or t > jump_at + 60:
            btn |= NES_INPUT_RIGHT
        if t == jump_at:
            btn |= NES_INPUT_A
        if t == jump_at + throw_delay:
            btn |= NES_INPUT_B
        nes.controller = btn
        arr = np.asarray(nes.step(), dtype=np.uint8)
        nes.controller = 0
        hx = int(nes[XL + HEAD_SLOT])
        hy = int(nes[YL + HEAD_SLOT])
        reds.append(red_count(arr, hx, hy))
        hps.append(int(nes[HP]))
        head_xy.append((hx, hy))
        # geometric: an axe (id 66) inside the head box
        for s in range(1, 14):
            if int(nes[ID + s]) == 66:
                ax, ay = int(nes[XL + s]), int(nes[YL + s])
                if abs(ax - hx) <= 14 and abs(ay - hy) <= 16:
                    hit_geom.append((t, s, ax, ay, hx, hy))
        if dump_buf is not None:
            for a in range(N_RAM):
                dump_buf[t, a] = nes[a]
    return np.array(reds), np.array(hps), hit_geom, dump_buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_bossface.state"))
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()

    nes = NES(ROM)
    combos = list(itertools.product([0, 20, 40, 60],        # run length
                                    [30, 50, 70, 90],       # jump moment
                                    [0, 6, 12, 18, 24]))    # throw delay
    rng = np.random.default_rng(7)
    if len(combos) > args.trials:
        combos = [combos[i] for i in rng.choice(len(combos), args.trials,
                                                False)]
    print(f"{len(combos)} combos x {args.frames} frames")

    base_red = None
    winners = []
    for i, (rl, ja, td) in enumerate(combos):
        reds, hps, geom, _ = play(nes, args.state, args.frames, rl, ja, td)
        if base_red is None:
            base_red = int(np.median(reds))
        spike = int(reds.max()) - base_red
        hp_drop = int(hps[0] - hps.min())
        if spike > 40 or hp_drop or geom:
            print(f"  trial{i} run={rl} jump={ja} throw=+{td}: "
                  f"red_max={int(reds.max())} (base {base_red}) "
                  f"hp {hps[0]}->{hps.min()} geom={geom[:4]}", flush=True)
            winners.append((spike, hp_drop, i, (rl, ja, td)))
    winners.sort(reverse=True)
    print("\n=== best trials ===")
    for spike, hp_drop, i, prm in winners[:10]:
        print(f"  trial{i} {prm} red_spike={spike} hp_drop={hp_drop}")

    if winners:
        spike, hp_drop, i, prm = winners[0]
        reds, hps, geom, buf = play(nes, args.state, args.frames, *prm,
                                    dump=True)
        flash = [t for t in range(1, len(reds))
                 if reds[t] - int(np.median(reds)) > 40]
        print(f"\nreplay trial{i} {prm}: flash frames={flash[:20]} "
              f"hp={hps.tolist()[:60]}")
        np.save(os.path.join(_ROOT, "outputs", "boss_hp5_hit.npy"), buf)
        print("dump -> outputs/boss_hp5_hit.npy")


if __name__ == "__main__":
    main()
