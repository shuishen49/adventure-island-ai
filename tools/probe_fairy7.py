"""Last-resort fairy probe: instead of moving the PLAYER onto the fairy
(whose physics the game overwrites every frame), move the FAIRY onto the
player.  Collision routines read both slots, so this should trigger.

For every attempt we snapshot the whole 2 KB of *real* RAM (nes[i], NOT
nes.save() -- that returns a 5382-byte save-state blob) and diff it, then
watch any changed byte for 240 frames to see if one of them counts down.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R
from env import ACTIONS, AdventureIslandEnv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAIRY_ID = 199
OID, ST = R.OBJ.id, R.OBJ.STRIDE
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_RJ = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump"][0]


def ram_snapshot(nes) -> np.ndarray:
    return np.frombuffer(bytes(nes[i] for i in range(2048)), dtype=np.uint8).copy()


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "1-3"
    area, rnd = (int(p) for p in name.split("-"))
    state = os.path.join(_ROOT, "saves", "stages", f"stage_{name}.state")
    env = AdventureIslandEnv(start_state=state, start_with_axe=True,
                             stage_index=(area - 1) * 4 + (rnd - 1))
    env.reset(seed=0)
    nes = env._nes

    attempts = 0
    for t in range(3000):
        if int(nes[R.DEATH]) != 0:
            nes[R.DEATH] = 0
        nes[R.STAMINA] = 10

        slot = -1
        for s in range(1, 16):
            if int(nes[OID + ST * s]) == FAIRY_ID:
                slot = s
                break

        if slot >= 0 and attempts < 3:
            attempts += 1
            print(f"\n=== 尝试 #{attempts}  决策 t={t}  slot={slot} ===")
            before = ram_snapshot(nes)
            pscore = int(nes[R.SCORE]) if hasattr(R, "SCORE") else -1
            plives = int(nes[R.LIVES])

            # park the fairy exactly on the player, every frame, for 30 frames
            for f in range(30):
                for off in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr):
                    nes[off + ST * slot] = nes[off + ST * 0]
                for off in (R.OBJ.y_lo, R.OBJ.y_fr):
                    nes[off + ST * slot] = nes[off + ST * 0]
                nes.controller = 0
                nes.step()
                gone = int(nes[OID + ST * slot]) != FAIRY_ID
                if gone:
                    print(f"  frame {f}: 精灵消失! (slot {slot} 不再是 199)")
                    break
            after = ram_snapshot(nes)
            nscore = int(nes[R.SCORE]) if hasattr(R, "SCORE") else -1
            print(f"  score {pscore} -> {nscore}   lives {plives} -> {int(nes[R.LIVES])}")
            print(f"  30帧后 slot{slot} id = {int(nes[OID + ST * slot])}")

            diff = np.flatnonzero(before != after)
            print(f"  RAM 变化字节数: {len(diff)}")

            # follow anything that JUMPED UP at the contact moment and then
            # ticks down -- that is an invincibility timer
            cands = [int(d) for d in diff if after[d] > before[d] + 20 and after[d] < 250]
            cands = cands[:12]
            if cands:
                traj = {c: [int(after[c])] for c in cands}
                for k in range(240):
                    nes[R.STAMINA] = 10
                    if int(nes[R.DEATH]) != 0:
                        nes[R.DEATH] = 0
                    nes.controller = 0
                    nes.step()
                    if k % 20 == 19:
                        for c in cands:
                            traj[c].append(int(nes[c]))
                for c in cands:
                    seq = traj[c]
                    drops = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
                    flag = "  <== 倒计时?" if drops >= len(seq) - 2 else ""
                    print(f"    ${c:04X}: {seq}{flag}")
            sys.stdout.flush()

        env.step(A_RJ if t % 26 in (0, 1) else A_RIGHT)

    print(f"\n总尝试次数: {attempts}")
    env.close()


if __name__ == "__main__":
    main()
