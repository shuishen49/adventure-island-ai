"""Where does the 1-4 agent actually die?  (evidence, not guesses)

eval_native says: 16 episodes, avg 2221 px, best 3819, but the MODE of the
distribution sits at 1850..1950 px -- i.e. most episodes never even reach
the boss arena at 3592 px.  Before we spend another 8 M steps we want to
know what kills them there: a pit they cannot jump, an enemy they cannot
kill, or an empty stamina meter.

For every episode we log the last 40 decisions (progress, camera, stamina,
player screen y) and classify the death:

    pit      player y drops well below the ground line at the death frame
             (ground is y~184; a fall into a hole keeps growing)
    starve   $0076 reached 0 (verified: death fires on that exact frame)
    enemy    an enemy slot is within 24 px of the player at the death frame
    other    none of the above

A few death frames are dumped as PNGs next to the log so we can eyeball
the terrain.

    python tools/diag_14_deaths.py [--episodes 16] [--greedy]
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
from env import (AdventureIslandEnv, OBS_DIM, N_ACTIONS,   # noqa: E402
                 MAX_STEPS)
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "runs", "ppo_14", "ppo_final.pt")
OUT = os.path.join(_ROOT, "outputs")

GROUND_Y = 184          # measured: standing sprites sit at y ~184
# Measured 2026-10-18 (standing still on 1-4): $0076 starts at 10 and the
# death flag goes up on the exact frame it reaches 0 (frame 2796).  So a
# REAL starvation death ends with stamina == 0 -- the old threshold of 8
# mislabelled 11 of 16 contact deaths as "starve".
STARVE_STAMINA = 1      # $0076 ticks down to 0 -> death by starvation


def player_y(nes) -> float:
    return R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))


def player_x(nes) -> float:
    return R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))


def nearby(nes, limit: int = 24):
    """(oid, dx, dy) for objects within ``limit`` px of the player."""
    px, py = player_x(nes), player_y(nes)
    out = []
    for s in range(1, 16):
        base = R.OBJ.id + R.OBJ.STRIDE * s
        oid = int(nes[base])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        if abs(ox - px) <= limit and abs(oy - py) <= 64:
            out.append((oid, int(ox - px), int(oy - py)))
    return out


def classify(hist: list[dict], nes) -> str:
    last = hist[-1]
    if last["y"] > GROUND_Y + 24:
        return "pit"
    if last["stam"] <= STARVE_STAMINA and all(
            h["stam"] <= STARVE_STAMINA + 4 for h in hist[-6:]):
        return "starve"
    if last["near"]:
        return "enemy"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--shots", type=int, default=6,
                    help="how many death frames to dump as PNG")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3,
                             clear_ends_episode=True)

    rows, shots = [], 0
    for e in range(args.episodes):
        o = env.reset(seed=1000 + e)
        hist, done, n = [], False, 0
        while not done and n < MAX_STEPS:
            batch = {"ram": torch.from_numpy(o["ram"][None]),
                     "img": torch.from_numpy(o["img"][None])}
            with torch.no_grad():
                lg, _ = net(batch)
            a = int(lg[0].argmax()) if args.greedy else int(
                torch.distributions.Categorical(logits=lg[0]).sample())
            o, r, done, info = env.step(a)
            n += 1
            nes = env._nes
            hist.append({"x": float(info.get("x_progress", 0.0)),
                         "cam": int(nes[R.CAMERA]),
                         "stam": int(nes[R.STAMINA]),
                         "y": player_y(nes),
                         "near": nearby(nes)})
            if len(hist) > 40:
                hist.pop(0)

        cause = classify(hist, env._nes) if hist else "empty"
        last = hist[-1] if hist else {"x": 0, "stam": 0, "y": 0, "near": []}
        cleared = bool(info.get("cleared", False))
        rows.append({"ep": e + 1, "x": float(info["x_progress"]),
                     "cause": "CLEAR" if cleared else cause,
                     "stam": last["stam"], "y": last["y"],
                     "near": last["near"][:3], "steps": n})

        if not cleared and shots < args.shots:
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(OUT, f"d14_ep{e + 1}.png"),
                            env._last_frame)
            shots += 1

        print(f"  ep{e + 1:2d}: {info['x_progress']:7.1f} px  "
              f"{n:4d} 决策  死因 {rows[-1]['cause']:<7}"
              f"  stamina={last['stam']:<3} y={last['y']:6.1f}  "
              f"贴身={last['near'][:3]}")

    xs = np.array([r["x"] for r in rows])
    print()
    print(f"平均 {xs.mean():.1f} px  中位 {np.median(xs):.1f}  "
          f"最好 {xs.max():.1f}  最差 {xs.min():.1f}")

    print("\n死因统计:")
    for c in ("pit", "starve", "enemy", "other", "CLEAR"):
        k = sum(1 for r in rows if r["cause"] == c)
        if k:
            at = [f"{r['x']:.0f}" for r in rows if r["cause"] == c]
            print(f"  {c:<7} {k:2d} 局   x = {', '.join(at)}")

    print("\n死亡位置分布 (200 px 一档):")
    lo, hi = 0, int(xs.max() // 200) + 1
    for b in range(lo, hi + 1):
        k = sum(1 for r in rows if b * 200 <= r["x"] < (b + 1) * 200)
        if k:
            print(f"  {b * 200:5d}-{b * 200 + 199:5d} px | "
                  f"{'#' * k} {k}")


if __name__ == "__main__":
    main()
