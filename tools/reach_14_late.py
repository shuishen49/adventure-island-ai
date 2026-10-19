"""Reach the last stretch of 1-4 WITHOUT cheats, and describe it.

Why no cheats: forcing object 199 (the invincibility puff) into a slot
makes 1-1 and 1-2 run to ~29000 px without ever clearing, so every
"1-4 is 25000 px long" number measured that way is an artefact of the
injection.  Poking DEATH is worse (it traps the game in its death
routine).  So this script plays honestly: run the policy until it gets
past `--target` px, then

  * save the frame as a state (so later probes can start there);
  * print the whole screen as ASCII;
  * print a magnified ASCII crop around every nearby object, because the
    boss is probably just "the big sprite" and I cannot view PNGs.

    python tools/reach_14_late.py --target 4400
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
from env import AdventureIslandEnv, N_ACTIONS, OBS_DIM, MAX_STEPS  # noqa
from train import ActorCritic                              # noqa: E402

RAMP = " .:-=+*#%@"


def ascii_art(frame, w=76, h=30, box=None):
    g = frame.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if box is not None:
        x0, y0, x1, y1 = box
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(g.shape[1], int(x1)), min(g.shape[0], int(y1))
        g = g[y0:y1, x0:x1]
        if g.size == 0:
            return "(crop empty)"
    H, W = g.shape
    ys = np.linspace(0, H, h + 1).astype(int)
    xs = np.linspace(0, W, w + 1).astype(int)
    for i in range(h):
        if ys[i + 1] <= ys[i]:
            ys[i + 1] = min(H, ys[i] + 1)
    for j in range(w):
        if xs[j + 1] <= xs[j]:
            xs[j + 1] = min(W, xs[j] + 1)
    return "\n".join(
        "".join(RAMP[min(len(RAMP) - 1,
                         int(np.nan_to_num(
                             g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean())
                             / 255.0 * len(RAMP)))]
                for j in range(w))
        for i in range(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="runs/ppo_14d/ppo_27648000.pt")
    ap.add_argument("--target", type=float, default=4400.0)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_late.state"))
    args = ap.parse_args()
    if not os.path.exists(args.checkpoint):
        import glob
        cands = sorted(glob.glob(os.path.join(
            _ROOT, "runs", "ppo_14d", "*.pt")))
        if not cands:
            raise SystemExit("no 14d checkpoint")
        args.checkpoint = cands[-1]
    os.makedirs(os.path.dirname(args.state), exist_ok=True)

    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"model={args.checkpoint}")

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    best_overall = 0.0
    for ep in range(args.episodes):
        o = env.reset(seed=1000 + ep)
        nes = env._nes
        n, hit = 0, False
        while n < MAX_STEPS:
            with torch.no_grad():
                a = int(net({"ram": torch.from_numpy(o["ram"][None]),
                             "img": torch.from_numpy(o["img"][None])}
                            )[0].argmax(-1)[0])
            o, r, done, info = env.step(a)
            n += 1
            p = info.get("best_progress", 0.0)
            best_overall = max(best_overall, p)
            if p >= args.target and not hit:
                hit = True
                with open(args.state, "wb") as fh:
                    fh.write(bytes(nes.save()))
                print(f"\n=== ep{ep}: reached {p:.0f} px at step {n} "
                      f"-> {args.state} ===")
                break
            if done:
                break
        if hit:
            break
        print(f"  ep{ep}: best {info.get('best_progress', 0):.0f} px "
              f"(overall {best_overall:.0f})", flush=True)

    if not os.path.exists(args.state):
        print(f"\nnever reached {args.target:.0f} px; "
              f"overall best {best_overall:.0f} px")
        return

    # ---- describe what is on screen at that moment --------------------
    nes = env._nes
    px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    camx = px - 96.0
    frame = np.asarray(nes.step(0) if False else nes.step(1),
                       dtype=np.uint8)
    print(f"\nplayer world=({px:.0f},{py:.0f}) cam0={camx:.0f} "
          f"$007A={int(nes[R.STAGE_COMPLETE])} "
          f"DEATH={int(nes[R.DEATH])} LIVES={int(nes[R.LIVES])}")
    print("\n--- full screen ---")
    print(ascii_art(frame))
    for s in range(16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        label = R.OBJ_KINDS.get(oid, ("?",))[0]
        sx = ox - camx
        print(f"\n--- slot{s} id{oid} {label} "
              f"dx{ox - px:+.0f} dy{oy - py:+.0f} screenx={sx:.0f} ---")
        if -40 < sx < 300:
            print(ascii_art(frame, w=48, h=20,
                            box=(sx - 26, oy - 22, sx + 26, oy + 22)))


if __name__ == "__main__":
    main()
