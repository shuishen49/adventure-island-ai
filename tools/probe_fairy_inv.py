"""Decisive test: is the player INVINCIBLE while object 199 (the white
puff that follows the player) is on stage?

Method: move an enemy slot onto the player for ~24 frames and watch the
death flag ($0072):
  - with 199 present  -> should stay 0 if the puff means invincibility
  - without 199       -> should set (sanity check that the poke works)

    python tools/probe_fairy_inv.py 1-3
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R
from env import ACTIONS, AdventureIslandEnv

OID, ST = R.OBJ.id, R.OBJ.STRIDE
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]

ENEMY_IDS = {9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25}


def find_enemy(nes):
    for s in range(1, 16):
        if int(nes[OID + ST * s]) in ENEMY_IDS:
            return s
    return -1


def poke_enemy_onto_player(nes, slot, frames=24) -> int:
    """Returns 1 if the death flag fired."""
    fired = 0
    for _ in range(frames):
        for off in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr):
            nes[off + ST * slot] = nes[off + ST * 0]
        for off in (R.OBJ.y_lo, R.OBJ.y_fr):
            nes[off + ST * slot] = nes[off + ST * 0]
        nes.controller = 0
        nes.step()
        if int(nes[R.DEATH]) != 0:
            fired = 1
            break
    return fired


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    area, rnd = (int(p) for p in name.split("-"))
    state = os.path.join(_ROOT := os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "saves", "stages", f"stage_{name}.state")
    env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    tested_with = tested_without = 0
    for t in range(4000):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        has199 = any(int(nes[OID + ST * s]) == 199 for s in range(16))
        es = find_enemy(nes)

        if es >= 0 and tested_with < 2 and has199:
            print(f"t={t}  199在场  敌人slot={es} id={int(nes[OID+ST*es])} -> 钉上去")
            r = poke_enemy_onto_player(nes, es)
            print(f"    死亡标志: {r}  (0=没死 => 无敌!)")
            tested_with += 1
            sys.stdout.flush()
        elif es >= 0 and tested_without < 2 and not has199 and tested_with >= 1:
            print(f"t={t}  199不在  敌人slot={es} id={int(nes[OID+ST*es])} -> 钉上去")
            r = poke_enemy_onto_player(nes, es)
            print(f"    死亡标志: {r}  (1=死了 => 对照组正常)")
            tested_without += 1
            sys.stdout.flush()
        if tested_with >= 2 and tested_without >= 2:
            break

        env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)

    print(f"\n完成: 无敌组 {tested_with} 次 / 对照组 {tested_without} 次")
    env.close()


if __name__ == "__main__":
    main()
