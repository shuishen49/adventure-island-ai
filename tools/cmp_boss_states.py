#!/usr/bin/env python
"""Why does the policy fight in one arena snapshot and die in another?

Two states can both show "the boss ~90 px ahead" and still be completely
different games: extra objects on screen (spiders), a different boss phase,
Higgins arriving mid-fall with 6 stamina instead of standing with 10, the
camera not yet locked.  This tool dumps what the object table actually says
at frame 0 and then replays the policy, printing the distance / hit trace.

usage:
  python tools/cmp_boss_states.py saves/stages/stage_1-4boss.state \
         saves/stages/stage_1-4gate.state --checkpoint runs/ppo_14l/X.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                          # noqa: E402
from env import AdventureIslandEnv, N_ACTIONS, OBS_DIM       # noqa: E402


def dump(tag: str, state: str, net, stage_idx: int, steps: int) -> None:
    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=stage_idx, start_state=state)
    o = env.reset(seed=500)
    nes = env._nes
    ID, XL, YL = R.OBJ.id, R.OBJ.x_lo, R.OBJ.y_lo
    px, py = int(nes[XL]), int(nes[YL])
    print(f"\n=== {tag}  ({os.path.basename(state)}) ===")
    print(f"  player: x={px} y={py}  stamina={int(nes[R.STAMINA])} "
          f"lives={int(nes[R.LIVES])} weapon={int(nes[R.WEAPON])} "
          f"camera={int(nes[R.CAMERA])} progress={env._cam.progress(nes):.0f}")
    objs = []
    for s in range(1, 16):
        oid = int(nes[ID + R.OBJ.STRIDE * s])
        if oid:
            objs.append((s, oid, int(nes[XL + R.OBJ.STRIDE * s]),
                         int(nes[YL + R.OBJ.STRIDE * s])))
    print("  objects (slot,id,x,y): " + ", ".join(
        f"[{s}]{oid}@{x},{y}" for s, oid, x, y in objs))
    head = [s for s, oid, _, _ in objs if oid == R.BOSS_HEAD_ID]
    body = [s for s, oid, _, _ in objs if oid == R.BOSS_BODY_ID]
    print(f"  head slot={head} body slot={body} "
          f"dx={objs[0][2] - px if head else '-'} "
          f"head y={[y for s, oid, _, y in objs if oid == R.BOSS_HEAD_ID]} "
          f"body y={[y for s, oid, _, y in objs if oid == R.BOSS_BODY_ID]}")
    # replay
    trace = []
    for n in range(steps):
        with torch.no_grad():
            a = int(net({"img": torch.as_tensor(o["img"][None]),
                         "ram": torch.as_tensor(o["ram"][None],
                                                dtype=torch.float32)}
                        )[0].argmax())
        o, _, done, info = env.step(a)
        slots = [s for s in range(1, 16)
                 if int(nes[ID + R.OBJ.STRIDE * s]) == R.BOSS_HEAD_ID]
        dx = (int(nes[XL + R.OBJ.STRIDE * slots[0]]) - int(nes[XL])
              if slots else None)
        trace.append((n, a, dx, int(nes[YL]), int(nes[R.STAMINA]),
                      env._boss_hit_count(), int(nes[R.DEATH])))
        if done:
            break
    for n, a, dx, y, st, hits, death in trace[::15]:
        print(f"    n={n:<4} act={a:<2} dx={dx if dx is None else dx:<5} "
              f"y={y:<4} stam={st:<3} hits={hits} death={death}")
    last = trace[-1]
    print(f"  ended: n={last[0]} dx={last[2]} hits={last[5]} "
          f"death={last[6]} done={done}")
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="+")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stage-idx", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()

    from train import ActorCritic
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    for st in args.states:
        dump(os.path.basename(st)[6:-6], st, net, args.stage_idx, args.steps)


if __name__ == "__main__":
    main()
