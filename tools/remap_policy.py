"""Carry an 8-action policy head over to the 10-action head.

WHY: v5 warm-started from the 378 k checkpoint but had to throw the whole
policy head away, because it grew from 8 rows to 10 when
``jump+throw`` / ``right+jump+throw`` were added.  A freshly random head
means the agent forgets "run right, jump, throw" and has to re-learn it
from scratch -- which is exactly why the window showed it wandering left.

The old rows 0..7 keep their exact weights.  The two new rows are seeded
with the average of the two old actions they combine, so the agent starts
out treating them as "probably useful" instead of never trying them:

    jump+throw        = mean(jump, throw)
    right+jump+throw  = mean(right+jump, right+throw)

    python tools/remap_policy.py <old.pt> <out.pt>
"""

from __future__ import annotations

import sys

import torch


def remap(ck: dict, old_n: int, new_n: int,
          mapping: dict[int, tuple[int, ...]]) -> dict:
    sd = dict(ck["model"])
    for key in ("policy.weight", "policy.bias"):
        w = sd[key]
        if w.shape[0] != old_n:
            raise SystemExit(f"{key} has {w.shape[0]} rows, expected {old_n}")
        out = w.new_empty(new_n, *w.shape[1:])
        for i in range(old_n):
            out[i] = w[i]
        for new_i, parts in mapping.items():
            out[new_i] = torch.stack([w[p] for p in parts]).mean(0)
        sd[key] = out
    ck["model"] = sd
    return ck


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo/ppo_00378880.pt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "runs/ppo_v6_init.pt"

    # action table (env.ACTIONS), index -> name
    #   0 noop        1 right       2 left        3 jump
    #   4 right+jump  5 left+jump   6 throw       7 right+throw
    #   8 jump+throw  9 right+jump+throw
    mapping = {8: (3, 6), 9: (4, 7)}

    ck = torch.load(src, map_location="cpu", weights_only=False)
    old_n = int(ck["model"]["policy.weight"].shape[0])
    new_n = 10
    print(f"{src}: policy head {old_n} rows -> {new_n} rows")
    print(f"  rows 0..{old_n - 1}: copied verbatim")
    for i, parts in mapping.items():
        print(f"  row {i}: mean(rows {parts})")
    ck = remap(ck, old_n, new_n, mapping)
    ck["steps"] = 0
    torch.save(ck, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
