"""Carry the ppo_12 (1-2) checkpoint over to the 1-3 observation.

Two things change between runs/ppo_12 and the next curriculum:

1.  A FIREBALL bit is inserted into the player block.  `$006D` turned out
    to be a weapon TYPE (0 none / 1 axe / 2 fireball -- measured with
    tools/probe_fireball.py: same object id 66 in the air, but reach
    270 px vs 186 px), and the new obs feeds it as a second 0/1 feature
    right after the has-weapon bit.  OBS_DIM grows 69 -> 70.

    Column mapping of the RAM block (old -> new):
        0..4   x, y, spd, grav, stamina   -> unchanged
        5      has-weapon                 -> new 5 (unchanged)
                                          -> new 6 = ZEROS (fireball bit)
        6..36  skate .. nearest-item      -> new 7..37
        37..68 stage one-hot             -> new 38..69 (see 2)

2.  The stage one-hot SEMANTICS change.  ppo_12 was trained with the pool
    in CLI order ("1-2:0.7,1-1:0.3"), so its one-hot meant
        old 0 = 1-2,  old 1 = 1-1,  old 2..31 = unused
    train.py now sorts the pool, so the new run means
        new 0 = 1-1,  new 1 = 1-2,  new 2 = 1-3,  new 3..31 = unused
    Carrying the old matrix over unpermuted would silently swap 1-1 and
    1-2.  We therefore copy old column 1 into new column 0, old column 0
    into new column 1, and zero new column 2 (1-3 is new).

    python tools/make_13_init.py runs/ppo_12/ppo_final.pt runs/ppo_13/ppo_init.pt
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from env import N_ACTIONS, N_STAGES, OBS_DIM  # noqa: E402
from train import ActorCritic  # noqa: E402

OLD_DIM = 69                    # the ppo_12 observation: no fireball bit and
                                # the stage one-hot still in CLI order.
                                # NOTE: kept as a literal on purpose -- the env
                                # has since grown to 71 (fireball 70, then the
                                # invincibility bit 71) and OBS_DIM - 1 would
                                # silently pick the wrong source layout.


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu", weights_only=False)
    old = ck["model"]

    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    new = net.state_dict()
    for k, v in new.items():
        if k in old and old[k].shape == v.shape:
            new[k] = old[k].clone()

    w = old["fuse.0.weight"]                       # [128, FLAT + 69]
    FLAT = w.shape[1] - OLD_DIM                    # image features (2334)
    assert (FLAT + OLD_DIM, ) == w.shape[1:], tuple(w.shape)
    out = torch.zeros(128, FLAT + OBS_DIM, dtype=w.dtype)

    # --- image features: unchanged -----------------------------------
    out[:, :FLAT] = w[:, :FLAT]
    o = FLAT                                       # old RAM cursor
    n = FLAT                                       # new RAM cursor

    # --- player block: insert the fireball bit ------------------------
    out[:, n + 0:n + 5] = w[:, o + 0:o + 5]        # x y spd grav stamina
    out[:, n + 5] = w[:, o + 5]                    # has-weapon
    #               n + 6 stays ZERO                   has-fireball (new)
    out[:, n + 6:n + 8] = w[:, o + 6:o + 8]        # skate, btn-A
    o += 8
    n += 9

    # --- the rest of the RAM block ------------------------------------
    rest = OLD_DIM - N_STAGES - 8                  # 16+4+9 = 29
    out[:, n:n + rest] = w[:, o:o + rest]
    o += rest
    n += rest

    # --- stage one-hot: remap old CLI order -> sorted order ------------
    # old: 0 = 1-2, 1 = 1-1        new: 0 = 1-1, 1 = 1-2, 2 = 1-3
    out[:, n + 0] = w[:, o + 1]                    # 1-1
    out[:, n + 1] = w[:, o + 0]                    # 1-2
    #     n + 2 stays ZERO                          # 1-3 (new stage)
    # old 2..31 were zero anyway -> new 3..31 zero too

    new["fuse.0.weight"] = out
    net.load_state_dict(new)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save({"model": net.state_dict(), "steps": 0,
                "args": ck.get("args", {}),
                "widened_from": src}, dst)
    print(f"wrote {dst}")
    print(f"  obs_dim {OLD_DIM} -> {OBS_DIM} (fireball bit inserted at "
          f"player[6], stage one-hot remapped 1-1<->1-2, 1-3 zeroed)")


if __name__ == "__main__":
    main()
