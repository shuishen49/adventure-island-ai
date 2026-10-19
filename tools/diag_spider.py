"""1-4: is the agent even THROWING at the spider that kills it?

diag_killer_id.py says 6 of 12 deaths end glued to id 56 (SPIDER, dx ~ +6).
Two very different fixes depending on why:

  (a) it never throws  -> shaping problem: give a per-step penalty for being
      inside a lethal enemy's contact box so PPO learns to shoot/retreat
  (b) it throws and the spider does not die -> projectile problem: the axe
      parabola flies over a target that is only 6 px away, so we must teach
      it to keep distance before throwing

This tool logs, per episode: action histogram, throws, projectile frames,
spider appearances/disappearances (kill proxy) and the dx of the spider in
the 60 frames before the death.

    python tools/diag_spider.py --checkpoint runs/ppo_14c/ppo_18677760.pt \
        --episodes 8 --greedy
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, OBS_DIM, N_ACTIONS,   # noqa: E402
                 MAX_STEPS, ACTIONS, NES_INPUT_B)
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "runs", "ppo_14", "ppo_final.pt")
SPIDER = 56


def slots(nes):
    """[(slot, oid, x, y)] for every live slot."""
    out = []
    for s in range(0, 16):
        base = R.OBJ.id + R.OBJ.STRIDE * s
        oid = int(nes[base])
        if oid == 0:
            continue
        out.append((s, oid, R.px(R.read_obj_world_x(nes, s)),
                    R.px(R.read_obj_world_y(nes, s))))
    return out


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--start-round", type=int, default=3)
    ap.add_argument("--stage-idx", type=int, default=3)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.start_round,
                             stage_index=args.stage_idx,
                             clear_ends_episode=True)

    tot_throw = tot_proj = tot_kill56 = tot_seen56 = 0
    acts = collections.Counter()
    dx_at_death, dx_ever = [], []
    trans = collections.Counter()      # (old id) -> (new id) for enemy slots

    for e in range(args.episodes):
        o = env.reset(seed=3000 + e)
        done, n = False, 0
        throw, proj = 0, 0
        seen56, kill56 = set(), 0
        spider_dx = []                     # dx of nearest spider, per frame
        info = {}
        prev_alive = set()
        prev_ids = {}
        while not done and n < MAX_STEPS:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax()) if args.greedy else int(
                torch.distributions.Categorical(logits=lg[0]).sample())
            acts[a] += 1
            if ACTIONS[a][0] & NES_INPUT_B:
                throw += 1
            o, r, done, info = env.step(a)
            n += 1
            nes = env._nes
            live = slots(nes)
            ids_now = {s: oid for (s, oid, _, _) in live}
            for s, oid in prev_ids.items():
                if oid in R.ENEMY_IDS and ids_now.get(s, 0) != oid:
                    trans[(oid, ids_now.get(s, 0))] += 1
            prev_ids = ids_now
            if any(oid == R.ID_PROJECTILE for (_, oid, _, _) in live):
                proj += 1
            px, _ = player_xy(nes)
            sp = [(ox - px, oy) for (_, oid, ox, oy) in live if oid == SPIDER]
            now = {s for (s, oid, _, _) in live if oid == SPIDER}
            for s in prev_alive - now:
                # slot freed:蜘蛛消失 (可能被斧头打死, 也可能只是出屏)
                kill56 += 1
            prev_alive = now
            seen56 |= now
            if sp:
                spider_dx.append(min(abs(d) for (d, _) in sp))
                dx_ever.append(min(sp)[0])
            else:
                spider_dx.append(None)

        last = [d for d in spider_dx[-60:] if d is not None]
        dx_at_death.extend(last)
        tot_throw += throw
        tot_proj += proj
        tot_kill56 += kill56
        tot_seen56 += len(seen56)
        print(f"  ep{e + 1:2d}: {info.get('best_progress', 0.0):7.1f} px  "
              f"{n:4d} 步  投掷 {throw:3d}  飞行帧 {proj:3d}  "
              f"蜘蛛槽 {len(seen56)}  蜘蛛消失 {kill56}  "
              f"死前蜘蛛|dx|={last[-1] if last else '-'}")

    print()
    print(f"合计: 投掷 {tot_throw}  飞行帧 {tot_proj}  "
          f"蜘蛛出现槽位 {tot_seen56}  蜘蛛消失 {tot_kill56}")
    print("动作分布:", dict(acts))
    if trans:
        print("\n敌人槽位 id 变化 (old -> new, 出现次数):")
        for (a_, b_), k in trans.most_common(20):
            na = R.OBJ_KINDS.get(a_, ("?", (0, 0, 0)))[0]
            nb = R.OBJ_KINDS.get(b_, ("空", (0, 0, 0)))[0] if b_ else "空"
            print(f"  {a_:<4}({na:<9}) -> {str(b_):<5}({nb:<9})  {k:3d} 次")
    else:
        print("\n没有任何敌人槽位发生 id 变化 -> 一局里一个敌人都没被打死")

    if dx_ever:
        a = np.array(dx_ever)
        print(f"蜘蛛相对玩家的 dx: 中位 {np.median(a):.0f}  "
              f"近距(|dx|<40)占比 {np.mean(np.abs(a) < 40):.0%}  "
              f"远距(|dx|>90)占比 {np.mean(np.abs(a) > 90):.0%}")


if __name__ == "__main__":
    main()
