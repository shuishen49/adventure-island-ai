"""Which object id actually kills the 1-4 runner?  (evidence, not guesses)

diag_14_deaths.py only looks at ids that are ALREADY in ENEMY_IDS, so it can
never discover a killer we have not catalogued yet -- e.g. the pink ground
monster the user keeps seeing around 1000-1600 px.  This tool is blind to the
catalogue: at every death it dumps EVERY non-zero slot within a contact box
of the player, whatever its id, and tallies the ids over the last few frames
(the killer sometimes despawns on the exact death frame).

    python tools/diag_killer_id.py --checkpoint runs/ppo_14c/ppo_18554880.pt \
        --episodes 12 --greedy

Prints, per episode: best progress, death frame contact list, then a global
tally "id -> how many deaths it was touching at".
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
from env import AdventureIslandEnv, OBS_DIM, N_ACTIONS, MAX_STEPS  # noqa: E402
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "runs", "ppo_14", "ppo_final.pt")

# contact box: the player sprite is ~16x24 px, so anything overlapping this
# box at the death frame is the prime suspect.
DX = 30
DY = 44


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))


def contact(nes, dx=DX, dy=DY):
    """[(oid, dx, dy)] for ANY live slot overlapping the player box."""
    px, py = player_xy(nes)
    out = []
    for s in range(0, 16):
        base = R.OBJ.id + R.OBJ.STRIDE * s
        oid = int(nes[base])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        if abs(ox - px) <= dx and abs(oy - py) <= dy:
            out.append((oid, int(ox - px), int(oy - py)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=12)
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

    tally = collections.Counter()
    tally_last = collections.Counter()
    rows = []

    for e in range(args.episodes):
        o = env.reset(seed=2000 + e)
        done, n = False, 0
        recent = []                       # contact lists of the last 8 frames
        info = {}
        while not done and n < MAX_STEPS:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax()) if args.greedy else int(
                torch.distributions.Categorical(logits=lg[0]).sample())
            o, r, done, info = env.step(a)
            n += 1
            c = contact(env._nes)
            recent.append(c)
            if len(recent) > 8:
                recent.pop(0)

        alive = recent[-1] if recent else []
        # ids seen in contact during the last 8 frames (killer may despawn)
        window = {oid for c in recent for (oid, _, _) in c}
        for oid in window:
            tally[oid] += 1
        for (oid, _, _) in alive:
            tally_last[oid] += 1

        cleared = bool(info.get("cleared", False))
        rows.append((e + 1, float(info.get("best_progress", 0.0)),
                     "CLEAR" if cleared else "death", alive))
        print(f"  ep{e + 1:2d}: {info.get('best_progress', 0.0):7.1f} px  "
              f"{n:4d} 步  {'通关' if cleared else '死亡'}  "
              f"死亡帧贴身={alive}  近8帧={sorted(window)}")

    xs = np.array([r[1] for r in rows])
    print()
    print(f"平均 {xs.mean():.1f} px  中位 {np.median(xs):.1f}  "
          f"最好 {xs.max():.1f}  最差 {xs.min():.1f}")

    print("\n死亡前 8 帧内贴过身的 id 统计 (出现局数):")
    for oid, k in tally.most_common():
        name = R.OBJ_KINDS.get(oid, ("?", (0, 0, 0)))[0]
        tag = "  <-- 已在 ENEMY_IDS" if oid in R.ENEMY_IDS else ""
        print(f"  id{oid:<4} {name:<10} {k:2d}/{args.episodes} 局{tag}")

    print("\n死亡帧本身仍在贴身的 id:")
    for oid, k in tally_last.most_common():
        name = R.OBJ_KINDS.get(oid, ("?", (0, 0, 0)))[0]
        print(f"  id{oid:<4} {name:<10} {k:2d} 局")


if __name__ == "__main__":
    main()
