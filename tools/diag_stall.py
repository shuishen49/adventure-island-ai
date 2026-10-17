"""Where exactly does the 1-3 policy stall, and what is it doing there?

The user reports the ppo_final agent getting stuck against a tall blue
brick wall on 1-3, holding RIGHT and never jumping until the vitality bar
(STA) drains to zero.  This tool localises the stall:

  * greedy rollout on 1-3,
  * progress is tracked with the same CameraTracker the reward uses,
  * "stalled" = no new best progress for STALL_AFTER decisions,
  * at the stall we dump: progress, player x/y, the last 24 actions,
    and every object slot within 100 px,
  * plus a raw frame snapshot (outputs/stall_epN.png) so the spot can be
    looked at directly.

    python tools/diag_stall.py runs/ppo_13/ppo_final.pt \
        --start-state saves/stages/stage_1-3.state --stage-idx 2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, ".")
_ROOT = os.getcwd()

from env import ACTIONS, AdventureIslandEnv, OBS_DIM, N_ACTIONS  # noqa: E402
from train import ActorCritic  # noqa: E402
import ram_map as R  # noqa: E402

OID, ST = R.OBJ.id, R.OBJ.STRIDE
STALL_AFTER = 120


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--stage-idx", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--greedy", action="store_true", default=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_state=args.start_state,
                             stage_index=args.stage_idx,
                             start_with_axe=True, clear_ends_episode=True)

    stall_xs = []
    for e in range(args.episodes):
        o = env.reset(seed=2000 + e)
        nes = env._nes
        best = env._cam.progress(nes)
        best_t = 0
        done, n = False, 0
        last_actions: list[str] = []
        stall = None
        timeline = []
        while not done and n < 4000:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax())
            o, r, done, info = env.step(a)
            n += 1
            last_actions.append(ACTIONS[a][1])
            last_actions = last_actions[-24:]
            prog = env._cam.progress(nes)
            timeline.append(prog)
            if prog > best + 0.5:
                best, best_t = prog, n
            elif n - best_t == STALL_AFTER:
                px_ = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
                py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
                near = []
                for s in range(1, 16):
                    oid = int(nes[OID + ST * s])
                    if oid == 0:
                        continue
                    dx = R.px(R.read_obj_world_x(nes, s)) - px_
                    dy = R.px(R.read_obj_world_y(nes, s)) - py
                    if abs(dx) < 120:
                        near.append((oid, round(dx), round(dy)))
                stall = (best, px_, py, near, list(last_actions))
                buf = np.asarray(nes.step(), dtype=np.uint8).copy()
                img = Image.fromarray(buf, mode="RGB")
                d = ImageDraw.Draw(img)
                d.rectangle([px_ - 10, py - 16, px_ + 10, py + 2],
                            outline=(60, 255, 60))
                out = os.path.join(_ROOT, "outputs", f"stall_ep{e + 1}.png")
                img.resize((512, 480), Image.NEAREST).save(out)
                break
        if stall:
            prog, px_, py, near, acts = stall
            stall_xs.append(prog)
            print(f"ep{e + 1}: 卡在 {prog:7.0f}px (屏x={px_:.0f} y={py:.0f}) "
                  f"停滞 {STALL_AFTER} 决策  体力={int(nes[R.STAMINA])}")
            print(f"      附近物体 {near}")
            print(f"      最近动作 {acts}")
        else:
            print(f"ep{e + 1}: 无停滞  最终 {info['x_progress']:7.0f}px "
                  f"({'过关' if info.get('cleared') else '死亡'})")
    print(f"\n停滞点汇总: {[round(v) for v in stall_xs]}")


if __name__ == "__main__":
    main()
