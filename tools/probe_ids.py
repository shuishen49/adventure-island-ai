"""Automatically classify every object id by playing randomly.

Method (causal experiment):
  Play many episodes with a random / run-right mix of inputs.  Whenever
  an object overlaps the player, snapshot the game state and compare it
  10 frames later:

      lives  dropped  -> lethal enemy
      stamina dropped -> obstacle (damage, no death)
      score  rose     -> pick-up (fruit / item)
      nothing         -> decoration / harmless

Also records how often each id appears, whether it moves, and whether it
sits in the air or on the ground.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cynes import NES, NES_INPUT_A, NES_INPUT_B  # noqa: E402
from cynes import NES_INPUT_LEFT, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, ROM  # noqa: E402
from visualize import boot  # noqa: E402

CONTACT_DX = 10.0
CONTACT_DY = 20.0
DELAY = 10


def play(seed: int, frames: int) -> dict:
    rng = np.random.default_rng(seed)
    nes = NES(ROM)
    boot(nes)
    start_lives = nes[R.LIVES]

    seen = Counter()            # id -> frames on screen
    air = Counter()             # id -> frames above y=140 (airborne)
    moving = Counter()          # id -> frames with |spd| > 0
    contacts = Counter()        # id -> contact events
    eff = defaultdict(Counter)  # id -> effect counter

    pending = []                # (due_frame, id, lives, sta, score)
    in_contact = set()          # slots currently overlapping the player
    hold, hold_until = 0, 0
    frame = 0

    while frame < frames:
        if frame >= hold_until:
            mode = rng.random()
            if mode < 0.55:                      # run right, hop sometimes
                hold = NES_INPUT_RIGHT
                if rng.random() < 0.5:
                    hold |= NES_INPUT_A
            elif mode < 0.7:                     # go left
                hold = NES_INPUT_LEFT
                if rng.random() < 0.4:
                    hold |= NES_INPUT_A
            elif mode < 0.85:                    # jump in place
                hold = int(NES_INPUT_A)
            elif mode < 0.95:                    # throw
                hold = int(NES_INPUT_B)
            else:                                # idle
                hold = 0
            hold_until = frame + int(rng.integers(6, 30))
        nes.controller = hold

        buf = nes.step()
        del buf
        frame += 1

        # ---- resolve pending contact effects ----
        if pending and pending[0][0] <= frame:
            due, oid, lv0, st0, sc0 = pending.pop(0)
            lv1, st1 = nes[R.LIVES], nes[R.STAMINA]
            sc1 = nes[R.SCORE + 7]
            if lv1 < lv0:
                eff[oid]["death"] += 1
            elif st1 < st0:
                eff[oid]["damage"] += 1
            if sc1 != sc0:
                eff[oid]["score"] += 1
            if lv1 == lv0 and st1 == st0 and sc1 == sc0:
                eff[oid]["none"] += 1

        # ---- scan objects ----
        p_x = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        p_y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        st = nes[R.STAMINA]
        lv = nes[R.LIVES]
        sc = nes[R.SCORE + 7]
        for s in range(1, R.N_OBJ_SLOTS):
            if s == R.OBJ.CAMERA_SLOT:   # slot 15 = scroll object, no sprite
                continue
            oid = nes[R.OBJ.id + s]
            if oid == 0:
                continue
            x = R.px(R.read_obj_world_x(nes, s))
            y = R.px(R.read_obj_world_y(nes, s))
            if not (0 <= x < 256 and 0 <= y < 240):
                continue
            seen[oid] += 1
            if y < 140:
                air[oid] += 1
            spd = nes[R.OBJ.spd_x + s]
            if spd != 0:
                moving[oid] += 1

            touching = (abs(x - p_x) < CONTACT_DX and abs(y - p_y) < CONTACT_DY)
            if touching and s not in in_contact:
                contacts[oid] += 1
                pending.append((frame + DELAY, oid, lv, st, sc))
            if touching:
                in_contact.add(s)
            elif s in in_contact:
                in_contact.discard(s)

    nes.controller = 0
    del nes
    return {"seen": seen, "air": air, "moving": moving,
            "contacts": contacts, "eff": {k: dict(v) for k, v in eff.items()},
            "start_lives": start_lives}


def merge(results):
    seen, air, moving, contacts = Counter(), Counter(), Counter(), Counter()
    eff = defaultdict(Counter)
    for r in results:
        seen.update(r["seen"])
        air.update(r["air"])
        moving.update(r["moving"])
        contacts.update(r["contacts"])
        for k, v in r["eff"].items():
            eff[k].update(v)
    return seen, air, moving, contacts, eff


def classify(oid, seen, air, moving, contacts, eff):
    n = seen[oid]
    e = eff[oid]
    death = e.get("death", 0)
    dmg = e.get("damage", 0)
    score = e.get("score", 0)
    if death:
        kind = "LETHAL-ENEMY"
    elif dmg:
        kind = "OBSTACLE (damage)"
    elif score:
        kind = "PICKUP (score)"
    else:
        kind = "unknown"
    if contacts[oid] == 0:
        kind += "  [no contact observed]"
    return kind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with Pool(args.workers) as pool:
        results = pool.starmap(play,
                               [(i, args.frames) for i in range(args.workers)])

    seen, air, moving, contacts, eff = merge(results)
    print(f"\n{'id':>4} {'frames':>8} {'air%':>5} {'mov%':>5} "
          f"{'hits':>6}  effects -> classification")
    for oid in sorted(seen, key=lambda k: -seen[k]):
        n = seen[oid]
        if n < 30:                       # noise / transient
            continue
        e = eff[oid]
        kind = classify(oid, seen, air, moving, contacts, eff)
        print(f"{oid:4d} {n:8d} {100 * air[oid] // max(n, 1):5d} "
              f"{100 * moving[oid] // max(n, 1):5d} {contacts[oid]:6d}  "
              f"{e}  -> {kind}")

    print("\nRAW (all ids seen, including rare):")
    print("  " + ", ".join(f"{k}({v})" for k, v in sorted(seen.items(),
                                                          key=lambda t: -t[1])))


if __name__ == "__main__":
    main()
