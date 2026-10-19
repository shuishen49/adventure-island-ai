"""Analyse a --record-ram session: find the boss HP byte.

Layout note (ram_map.OBJ): the object table is FIELD-MAJOR --
    id      at 0x564 + slot
    x_fr    at 0x574 + slot      (each field block is 0x10 apart)
    x_lo    at 0x584 + slot
    ...
so a slot's 16 fields are spread 0x10 apart, not contiguous.  The recorded
window is 0x00-0xFF plus 0x500-0x6FF.

For every frame we locate the boss head (id 198) and body (id 67) and dump
each of their 16 field values over time.  A field that only ever goes DOWN
while the user was landing hits is the HP byte.

    python tools/analyze_manual.py outputs/manual_ram.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402

OBJ_BASE = 0x500
N_FIELD = 16
BOSS_HEAD = 198
BOSS_BODY = 67


def field_index(field_addr: int, slot: int) -> int:
    return field_addr - OBJ_BASE + slot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("--max-changes", type=int, default=40)
    args = ap.parse_args()

    arr = np.load(args.npy)
    T = len(arr)
    print(f"{T} frames, {arr.shape[1]} bytes/frame")

    id_base = int(R.OBJ.id, 16) if isinstance(R.OBJ.id, str) else int(R.OBJ.id)
    field_addrs = [OBJ_BASE + 0x10 * k for k in range(N_FIELD)]

    # which frames actually have the boss on screen?
    head_mask = np.zeros(T, dtype=bool)
    body_mask = np.zeros(T, dtype=bool)
    head_slot = np.full(T, -1, dtype=np.int16)
    for s in range(16):
        col = arr[:, id_base - OBJ_BASE + s]
        m = col == BOSS_HEAD
        head_mask |= m
        head_slot[m] = s
        body_mask |= col == BOSS_BODY
    print(f"boss HEAD on screen: {int(head_mask.sum())} frames "
          f"({head_mask.mean() * 100:.1f}%)")
    print(f"boss BODY on screen: {int(body_mask.sum())} frames "
          f"({body_mask.mean() * 100:.1f}%)")
    if not head_mask.any():
        print("the boss never appeared in this recording")
        return

    # slot stability: does the head stay in one slot?
    slots = head_slot[head_mask]
    vals, cnt = np.unique(slots, return_counts=True)
    print("head slot histogram:", dict(zip(vals.tolist(), cnt.tolist())))

    for name, want in (("HEAD(198)", BOSS_HEAD), ("BODY(67)", BOSS_BODY)):
        print(f"\n===== {name} =====")
        for k in range(N_FIELD):
            addr = field_addrs[k]
            idx = addr - OBJ_BASE          # + slot added per frame
            col = None
            for s in range(16):
                c = arr[:, idx + s]
                m = (arr[:, id_base - OBJ_BASE + s] == want)
                if not m.any():
                    continue
                cur = np.where(m, c.astype(np.int16), np.int16(-1))
                col = cur if col is None else np.maximum(col, cur)
            if col is None:
                continue
            v = col[col >= 0].astype(np.int16)
            if len(v) < 5:
                continue
            changes = [f"{int(v[i])}->{int(v[i + 1])}"
                       for i in range(len(v) - 1) if v[i] != v[i + 1]]
            if not changes:
                continue
            d = np.diff(v)
            d = d[d != 0]
            mono_down = bool((d < 0).all()) if len(d) else False
            flag = "  <== ONLY DECREASES" if mono_down else ""
            print(f"  field ${addr:04X}: uniq={len(np.unique(v))} "
                  f"changes={len(changes)}{flag}")
            print(f"      {' '.join(changes[:args.max_changes])}")

    # did anything that looks like a stage clear happen?
    for nm, a in (("STAGE_COMPLETE 0x7A", 0x7A), ("DEATH 0x72", 0x72),
                  ("LIVES 0x3F", 0x3F), ("STAMINA 0x76", 0x76)):
        if a < 0x100:
            col = arr[:, a]
            ch = [f"{int(col[i])}->{int(col[i + 1])}"
                  for i in range(len(col) - 1) if col[i] != col[i + 1]]
            print(f"\n{nm}: {len(ch)} changes -> "
                  f"{' '.join(ch[:20])}")


if __name__ == "__main__":
    main()
