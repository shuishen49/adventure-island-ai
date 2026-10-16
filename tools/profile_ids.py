"""Build a profile of each object id: which slots it uses, how it sits
relative to the player, whether it moves, and what happens on contact
(tighter contact test + longer observation window).
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cynes import NES, NES_INPUT_A, NES_INPUT_B  # noqa: E402
from cynes import NES_INPUT_LEFT, NES_INPUT_RIGHT  # noqa: E402

import ram_map as R  # noqa: E402
from env import ROM  # noqa: E402
from visualize import boot  # noqa: E402

CONTACT_DX, CONTACT_DY, DELAY = 8.0, 14.0, 18


def play(seed: int, frames: int) -> dict:
    rng = np.random.default_rng(seed)
    nes = NES(ROM)
    boot(nes)

    slots = defaultdict(Counter)   # id -> Counter(slot)
    dxs = defaultdict(list)        # id -> recent dx samples
    dys = defaultdict(list)
    spds = defaultdict(Counter)
    frames_n = Counter()
    near_player = Counter()        # frames within 12 px of the player
    eff = defaultdict(Counter)
    pending, in_contact = [], set()

    hold, hold_until, frame = 0, 0, 0
    while frame < frames:
        if frame >= hold_until:
            m = rng.random()
            if m < 0.55:
                hold = NES_INPUT_RIGHT | (NES_INPUT_A if rng.random() < .5 else 0)
            elif m < 0.72:
                hold = NES_INPUT_LEFT | (NES_INPUT_A if rng.random() < .4 else 0)
            elif m < 0.85:
                hold = int(NES_INPUT_A)
            elif m < 0.93:
                hold = int(NES_INPUT_B)
            else:
                hold = 0
            hold_until = frame + int(rng.integers(6, 26))
        nes.controller = hold
        nes.step()
        frame += 1

        while pending and pending[0][0] <= frame:
            _, oid, lv0, st0, sc0 = pending.pop(0)
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

        p_x = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        p_y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        st, lv, sc = nes[R.STAMINA], nes[R.LIVES], nes[R.SCORE + 7]
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
            frames_n[oid] += 1
            slots[oid][s] += 1
            spds[oid][nes[R.OBJ.spd_x + s]] += 1
            dx, dy = x - p_x, y - p_y
            if len(dxs[oid]) < 4000:
                dxs[oid].append(dx)
                dys[oid].append(dy)
            if abs(dx) < 12:
                near_player[oid] += 1
            hit = abs(dx) < CONTACT_DX and abs(dy) < CONTACT_DY
            if hit and s not in in_contact:
                pending.append((frame + DELAY, oid, lv, st, sc))
                in_contact.add(s)
            elif not hit and s in in_contact:
                in_contact.discard(s)

    nes.controller = 0
    del nes
    return {"frames": frames_n, "slots": {k: dict(v) for k, v in slots.items()},
            "dx": {k: np.array(v) for k, v in dxs.items()},
            "dy": {k: np.array(v) for k, v in dys.items()},
            "spd": {k: dict(v) for k, v in spds.items()},
            "near": near_player, "eff": {k: dict(v) for k, v in eff.items()}}


def main() -> None:
    with Pool(4) as pool:
        res = pool.starmap(play, [(i, 80000) for i in range(4)])

    frames, slots, near, eff = Counter(), defaultdict(Counter), Counter(), \
        defaultdict(Counter)
    dx, dy, spd = defaultdict(list), defaultdict(list), defaultdict(Counter)
    for r in res:
        frames.update(r["frames"])
        for k, v in r["slots"].items():
            slots[k].update(v)
        for k, v in r["dx"].items():
            dx[k].append(v)
        for k, v in r["dy"].items():
            dy[k].append(v)
        for k, v in r["spd"].items():
            spd[k].update(v)
        near.update(r["near"])
        for k, v in r["eff"].items():
            eff[k].update(v)

    print(f"{'id':>4} {'frames':>8} {'slots':<12} {'dx_med':>7} {'dy_med':>7} "
          f"{'mov%':>5} {'near%':>5}  effects")
    for oid in sorted(frames, key=lambda k: -frames[k]):
        if frames[oid] < 2000:
            continue
        ddx = np.concatenate(dx[oid]) if dx[oid] else np.array([0.0])
        ddy = np.concatenate(dy[oid]) if dy[oid] else np.array([0.0])
        tot = sum(spd[oid].values())
        mov = 100 * (tot - spd[oid].get(0, 0)) / max(tot, 1)
        top_slots = ",".join(str(s) for s, _ in
                             sorted(slots[oid].items(),
                                    key=lambda t: -t[1])[:4])
        print(f"{oid:4d} {frames[oid]:8d} {top_slots:<12} "
              f"{np.median(ddx):7.1f} {np.median(ddy):7.1f} {mov:5.0f} "
              f"{100 * near[oid] // max(frames[oid], 1):5d}  {dict(eff[oid])}")


if __name__ == "__main__":
    main()
