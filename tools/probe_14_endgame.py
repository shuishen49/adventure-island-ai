"""Where does 1-4 actually end?

The dumb "hold right and hop" bot stalls at ~4177 px.  Use the trained
policy (which reaches 4632-4872 px) WITH the invincibility cheat so it
cannot die, and see how far the level goes and what sits at the end.

    python tools/probe_14_endgame.py --checkpoint runs/ppo_14c/ppo_final.pt
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
from env import (AdventureIslandEnv, N_ACTIONS, MAX_STEPS,  # noqa: E402
                 OBS_DIM, ROM)
from train import ActorCritic                              # noqa: E402

RAMP = " .:-=+*#%@"


def ascii_art(frame: np.ndarray, w: int = 60, h: int = 26) -> str:
    g = frame.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
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


def ids(nes):
    return sorted({(s, int(nes[R.OBJ.id + R.OBJ.STRIDE * s]))
                   for s in range(16)
                   if int(nes[R.OBJ.id + R.OBJ.STRIDE * s])})


def force_fairy(nes) -> None:
    """Keep object id 199 (the invincibility puff) in a spare slot.

    tools/probe_fairy_inv.py proved the puff makes contact harmless, so
    this is real invincibility -- unlike poking DEATH (which traps the
    game in its death routine) or refilling lives (which lets respawns
    corrupt the camera-travel progress measure).
    """
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 199:
            return
    for s in range(1, 16):
        off = R.OBJ.STRIDE * s
        if int(nes[R.OBJ.id + off]) == 0:
            nes[R.OBJ.id + off] = 199
            for f in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr,
                      R.OBJ.y_lo, R.OBJ.y_fr):
                nes[f + off] = nes[f]
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ppo_14c/ppo_final.pt")
    ap.add_argument("--max-frames", type=int, default=40000)
    ap.add_argument("--round", type=int, default=3)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--cheat", action="store_true", default=True)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    want = ck["model"]["fuse.0.weight"].shape[1]
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    got = net.fuse[0].weight.shape[1]
    if got != want:
        raise SystemExit(f"obs dim mismatch: env {got} vs ckpt {want}")
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"model={args.checkpoint} fused={want}")

    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.round,
                             stage_index=args.stage)
    o = env.reset(seed=0)
    nes = env._nes
    cam = env._cam

    best = -1.0
    stall = 0
    for f in range(args.max_frames):
        if args.cheat:
            # NOTE: forcing DEATH=0 makes the game re-enter the death
            # routine every frame and the runner never respawns (measured:
            # stuck at 5182 px for 1500 frames, LIVES frozen).  Refilling
            # stamina/lives instead lets deaths resolve normally.
            force_fairy(nes)
            nes[R.STAMINA] = 10
            nes[R.WEAPON] = max(1, int(nes[R.WEAPON]))
            nes[R.LIVES] = 9
        b = {"ram": torch.from_numpy(o["ram"][None]),
             "img": torch.from_numpy(o["img"][None])}
        with torch.no_grad():
            a = int(net(b)[0].argmax(-1)[0])
        o, r, done, info = env.step(a)
        img = o["img"]
        frame = img[-1] if img.shape[0] <= 4 else img[..., -1]
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.5 \
            else frame.astype(np.uint8)
        p = info.get("best_progress", cam.progress(nes))
        if p > best + 0.5:
            best = p
            stall = 0
        else:
            stall += 1
        if f % 1200 == 0 and f:
            print(ascii_art(frame), flush=True)
        if f % 600 == 0:
            print(f"  f={f:6d} prog={p:7.0f} best={best:7.0f} "
                  f"wx={R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)):8.0f} "
                  f"cam={int(nes[R.CAMERA]):3d} $007A="
                  f"{int(nes[R.STAGE_COMPLETE])} "
                  f"ids={[v for _, v in ids(nes)]}", flush=True)
        if int(nes[R.STAGE_COMPLETE]) != 0 and int(nes[R.DEATH]) == 0:
            print(f"  >> STAGE_COMPLETE with DEATH==0 at f={f} "
                  f"prog={p:.0f}", flush=True)
            print(ascii_art(frame))
            break
        if stall > 1500:
            print(f"  >> stalled at {best:.0f} for 1500 frames (f={f})",
                  flush=True)
            print(f"     cam={int(nes[R.CAMERA])} $007A="
                  f"{int(nes[R.STAGE_COMPLETE])} "
                  f"DEATH={int(nes[R.DEATH])} LIVES={int(nes[R.LIVES])} "
                  f"$0037={int(nes[R.AREA])} $0038={int(nes[R.ROUND])}")
            print(f"     ids={ids(nes)}")
            print(ascii_art(frame))
            # what is near the player?
            try:
                near = env._active_enemies()
            except Exception:
                near = []
            print(f"     nearby: {near}")
            break
    else:
        print(f"  ran {args.max_frames} frames, best={best:.0f}")
        print(ascii_art(frame))


if __name__ == "__main__":
    main()
