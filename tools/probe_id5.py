"""id 5 is now the main killer on 1-4 -- what is it?

tools/diag_killer_id.py on ppo_14c (12 greedy episodes): average 4241 px,
best 4872 px -- the deaths moved to the END of the stage and 6 of 12 end
glued to **id 5** (dx 2..6, dy 0..9, i.e. overlapping the player).  id 5
is NOT in ENEMY_IDS, so right now the policy literally cannot see it.

Three questions, answered by experiment (no guessing):
  1. does it move, and how?
  2. is contact with it lethal?   (walk into it with the cheat OFF)
  3. can an axe destroy it?       (throw at it, watch the slot)

Also prints the frame as ASCII art (I cannot view PNGs) with the id-5
object marked, so we can tell what it looks like.

    python tools/probe_id5.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,      # noqa: E402
                 NES_INPUT_A, NES_INPUT_B)

RAMP = " .:-=+*#%@"
TARGET = 5


def ascii_art(frame: np.ndarray, w: int = 78, h: int = 32) -> str:
    g = frame.astype(np.float32).mean(axis=2)
    H, W = g.shape
    ys = np.linspace(0, H, h + 1).astype(int)
    xs = np.linspace(0, W, w + 1).astype(int)
    return "\n".join(
        "".join(RAMP[min(len(RAMP) - 1,
                         int(g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
                             / 255.0 * len(RAMP)))]
                for j in range(w))
        for i in range(h))


def find(nes, oid):
    """[(slot, x, y)] of every slot holding ``oid``."""
    out = []
    for s in range(1, 16):
        if int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) == oid:
            out.append((s, R.px(R.read_obj_world_x(nes, s)),
                        R.px(R.read_obj_world_y(nes, s))))
    return out


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))


def main() -> None:
    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam
    snap = None
    best, stall = 0.0, 0
    for f in range(60 * 60 * 10):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if f % 26 < 4 else 0)
        arr = np.asarray(nes.step(), dtype=np.uint8)
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        hits = find(nes, TARGET)
        if hits:
            px, py = player_xy(nes)
            slot, ox, oy = hits[0]
            if 10 < ox - px < 140:               # ahead of us, in range
                snap = nes.save()
                print(f"id5 在前方: f={f} progress={p:.0f} "
                      f"slot={slot} 玩家=({px:.0f},{py:.0f}) "
                      f"id5=({ox:.0f},{oy:.0f}) dx={ox - px:.0f} "
                      f"dy={oy - py:.0f}", flush=True)
                print("\n--- 画面 ---")
                print(ascii_art(arr))
                print("-----------\n", flush=True)
                break
        if stall > 45 * 60:
            print(f"卡住 {best:.0f} px, 没在射程内遇到 id5", flush=True)
            return

    # --- 1) does it move? --------------------------------------------
    traj = []
    for f in range(240):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes.controller = 0
        nes.step()
        h = find(nes, TARGET)
        if h:
            traj.append((h[0][1], h[0][2]))
    if traj:
        a = np.array(traj)
        print(f"站着不动 240 帧: id5 x {a[:, 0].min():.0f}..{a[:, 0].max():.0f}"
              f" (位移 {a[:, 0].max() - a[:, 0].min():.0f} px)  "
              f"y {a[:, 1].min():.0f}..{a[:, 1].max():.0f} "
              f"(位移 {a[:, 1].max() - a[:, 1].min():.0f} px)")

    # --- 2) lethal on contact?  (cheat OFF) --------------------------
    env2 = AdventureIslandEnv(start_with_axe=True, start_round=3,
                              stage_index=3)
    env2._reset_internal_state()
    env2._nes.load(np.frombuffer(snap, dtype=np.uint8).copy())
    env2._cam = R.CameraTracker(env2._nes)
    n2 = env2._nes
    died_at = None
    for f in range(600):
        n2.controller = NES_INPUT_RIGHT          # walk into it
        n2.step()
        if n2[R.DEATH] != 0 and n2[R.STAGE_COMPLETE] == 0:
            died_at = f
            break
    print(f"关掉作弊后向右撞上去: "
          f"{'第 %d 帧死亡 -> 接触致命' % died_at if died_at is not None else '600 帧内没死'}")

    # --- 3) can an axe kill it? --------------------------------------
    env3 = AdventureIslandEnv(start_with_axe=True, start_round=3,
                              stage_index=3)
    env3._reset_internal_state()
    env3._nes.load(np.frombuffer(snap, dtype=np.uint8).copy())
    env3._cam = R.CameraTracker(env3._nes)
    n3 = env3._nes
    gone_at, alive_at_end = None, False
    for f in range(900):
        n3[R.DEATH] = 0
        n3[R.STAMINA] = 10
        n3[R.WEAPON] = 1
        n3.controller = NES_INPUT_B | (NES_INPUT_A if f % 14 < 3 else 0)
        n3.step()
        if not find(n3, TARGET) and gone_at is None:
            gone_at = f
    alive_at_end = bool(find(n3, TARGET))
    print(f"狂扔斧头 900 帧: "
          f"{'id5 在第 %d 帧消失 -> 可击杀' % gone_at if gone_at is not None else ('900 帧后仍在场 -> 打不死' if alive_at_end else 'id5 中途消失过')}")


if __name__ == "__main__":
    main()
