"""Can an axe actually kill the 1-4 spider (id 56)?

diag_killer_id says 6 of 12 deaths end glued to a spider, and diag_spider
says the agent throws ~115 axes per episode -- so it is definitely shooting.
Two possibilities remain:

  (a) the axe never gets near the spider      -> aiming / distance problem
  (b) the axe flies right through the spider  -> the spider is immune and
      the only correct policy is to dodge it, which means we must add a
      proximity-danger shaping term instead of trusting kills

Slot ids cannot answer this (slots get recycled, see diag_spider), so we
track each spider as an ENTITY by position continuity and log the minimum
distance any projectile ever reached before the entity vanished.

    python tools/probe_spider_killable.py --checkpoint runs/ppo_14c/ppo_*.pt \
        --episodes 6 --greedy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import AdventureIslandEnv, OBS_DIM, N_ACTIONS, MAX_STEPS  # noqa: E402
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "runs", "ppo_14", "ppo_final.pt")
SPIDER = 56
MATCH_R = 40.0      # same entity if it moved less than this between frames


def objs(nes):
    """[(oid, x, y)] for every live slot (slot 0 = player excluded later)."""
    out = []
    for s in range(1, 16):
        base = R.OBJ.id + R.OBJ.STRIDE * s
        oid = int(nes[base])
        if oid == 0:
            continue
        out.append((oid, R.px(R.read_obj_world_x(nes, s)),
                    R.px(R.read_obj_world_y(nes, s))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3, clear_ends_episode=True)

    vanished_near, vanished_far = [], []      # min projectile dist at vanish
    n_entities = 0
    closest_alive = []                        # min dist while spider alive

    for e in range(args.episodes):
        o = env.reset(seed=5000 + e)
        done, n = False, 0
        ents = {}          # eid -> {"x","y","min_d","frames"}
        next_id = 0
        while not done and n < MAX_STEPS:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax()) if args.greedy else int(
                torch.distributions.Categorical(logits=lg[0]).sample())
            o, r, done, _info = env.step(a)
            n += 1
            live = objs(env._nes)
            shots = [(x, y) for (oid, x, y) in live if oid == R.ID_PROJECTILE]
            spiders = [(x, y) for (oid, x, y) in live if oid == SPIDER]

            used = set()
            for eid, st in list(ents.items()):
                best, bd = None, 1e9
                for (x, y) in spiders:
                    if id(x) in used:
                        continue
                    d = ((x - st["x"]) ** 2 + (y - st["y"]) ** 2) ** .5
                    if d < bd:
                        best, bd = (x, y), d
                if best is None or bd > MATCH_R:
                    # entity gone: was a projectile on top of it?
                    if st["min_d"] < 24:
                        vanished_near.append(st["min_d"])
                    else:
                        vanished_far.append(st["min_d"])
                    closest_alive.append(st["min_d"])
                    del ents[eid]
                    continue
                st["x"], st["y"] = best
                st["frames"] += 1
                for (sx, sy) in shots:
                    d = ((sx - best[0]) ** 2 + (sy - best[1]) ** 2) ** .5
                    st["min_d"] = min(st["min_d"], d)
            seen = {id(x) for st in ents.values() for _ in [0]}
            for (x, y) in spiders:
                if any(abs(x - st["x"]) < 1e-6 and abs(y - st["y"]) < 1e-6
                       for st in ents.values()):
                    continue
                ents[next_id] = {"x": x, "y": y, "min_d": 1e9, "frames": 1}
                next_id += 1
                n_entities += 1

        print(f"  ep{e + 1:2d}: 追踪到蜘蛛实体 {next_id} 个")

    print(f"\n蜘蛛实体总数 {n_entities}")
    if closest_alive:
        a = np.array(closest_alive)
        print(f"消失时最近斧头距离: 中位 {np.median(a):.1f} px  "
              f"<24px 的有 {len(vanished_near)} 次 / 共 {len(a)} 次")
        print(f"  贴着斧头消失(<24px): {[round(x, 1) for x in vanished_near][:20]}")
    else:
        print("没有任何蜘蛛实体消失过 (可能一直活到死亡/或追踪半径太小)")

    if vanished_near:
        print("\n结论: 斧头贴到蜘蛛后蜘蛛消失 -> 蜘蛛可以被打死, "
              "问题在瞄准/距离")
    elif closest_alive and np.median(closest_alive) < 24:
        print("\n结论: 斧头经常贴身飞过但蜘蛛不消失 -> 蜘蛛疑似免疫, "
              "应改为躲避(加贴身危险惩罚)")
    else:
        print("\n结论: 斧头几乎没接近过蜘蛛 -> 需要在更远的距离先手投掷")


if __name__ == "__main__":
    main()
