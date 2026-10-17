"""Evaluate a trained checkpoint over N deterministic episodes."""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from cynes import NES

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, push_img, reset_cam
from train import ActorCritic
from visualize import boot


def run_episode(net, nes, max_frames=4500, greedy=True,
                start_round=0, start_state=None, stage_idx=0,
                stop_on_death=True, wait_picture=True, start_axe=True):
    """One life: play from ``start_state`` until death / clear / frame cap.

    ``stop_on_death`` mirrors the training env (an episode ends the moment
    a life is lost), so the numbers here are comparable with the
    ``mean_x_progress`` column of runs/*/progress.csv.
    """
    boot(nes, start_round=start_round, start_state=start_state)
    # env.reset() arms the axe on every episode (start_with_axe=True is the
    # training default), and the policy is NOT invariant to it: on 1-2 the
    # 8M checkpoint averages 5961 px with the axe and 308 px without it
    # -- it dies at ~230 px every single time (measured 2026-10-17).  Any
    # harness that starts from a save state must write the flag too.
    if start_axe:
        nes[R.WEAPON] = 1
    reset_cam(nes)          # drops the stale camera AND the frame stack
    if wait_picture:
        # the PPU is off for ~180 frames after a stage load (black screen)
        f0 = None
        for _ in range(400):
            f0 = np.asarray(nes.step())
            if f0.mean() > 30:
                break
        # env.reset() seeds the 2-frame stack with the first frame; do the
        # same here or the net sees a zero channel on its first decisions.
        if f0 is not None:
            push_img(nes, f0)
            push_img(nes, f0)   # 2-deep stack, both channels = first frame
    cam = R.CameraTracker(nes)
    action = 0
    max_x = 0.0
    deaths = 0
    start_lives = nes[R.LIVES]
    last_lives = start_lives
    fruits = 0
    last_score = nes[0x06A1]
    last_weapon = nes[R.WEAPON]
    axes = 0
    cleared = False
    clear_x = 0.0

    for i in range(max_frames):
        if i % 4 == 0:
            with torch.no_grad():
                logits = net(obs_tensor(nes, stage_idx))[0]
            action = int(logits.argmax()) if greedy else int(
                torch.distributions.Categorical(logits=logits).sample())
        nes.controller = ACTIONS[action][0]
        buf = np.asarray(nes.step(), dtype=np.uint8)
        # env.step() runs frame_skip=4 NES frames per decision and pushes
        # ONLY the last one, so the net is trained on a stack whose two
        # channels are 4 game frames apart.  Pushing every frame here would
        # feed it a 1-frame gap -- a silent train/test mismatch that cost us
        # a bogus "230 px" measurement (2026-10-17).
        if i % 4 == 3:
            push_img(nes, buf)
        p = cam.progress(nes)
        if p > max_x:
            max_x = p
        lv = nes[R.LIVES]
        if lv < last_lives:
            deaths += 1
            last_lives = lv
        if nes[R.STAGE_COMPLETE] != 0:
            cleared = True
            clear_x = max_x
        sc = nes[0x06A1]
        if sc != last_score:
            fruits += 1
            last_score = sc
        w = nes[R.WEAPON]
        if w > last_weapon:
            axes += 1
        last_weapon = w
        if cleared or (stop_on_death and deaths):
            break
        if lv == 0:
            break
    nes.controller = 0
    return max_x, deaths, fruits, axes, cleared, clear_x, i + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_final.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=4500)
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--start-state", default=None)
    ap.add_argument("--stage-idx", type=int, default=0,
                    help="one-hot stage tail: 0 = 1-1, 1 = 1-2, ...")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--no-axe", action="store_true",
                    help="do NOT pre-equip the axe (training does)")
    ap.add_argument("--full-life", action="store_true",
                    help="keep going until game over instead of 1st death")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"loaded {args.checkpoint}  ({ck.get('steps', '?')} steps)")

    nes = NES(ROM)
    xs, ds, fs, ax = [], [], [], []
    nclear = 0
    where = args.start_state or f"round {args.start_round + 1}"
    print(f"stage: {where}   one-hot idx {args.stage_idx}")
    for e in range(args.episodes):
        x, d, f, a, cl, cx, nfr = run_episode(
            net, nes, args.max_frames, greedy=not args.stochastic,
            start_round=args.start_round, start_state=args.start_state,
            stage_idx=args.stage_idx, stop_on_death=not args.full_life,
            start_axe=not args.no_axe)
        xs.append(x)
        ds.append(d)
        fs.append(f)
        ax.append(a)
        nclear += int(cl)
        print(f"  ep{e+1:2d}: 最远 {x:7.1f} px  帧 {nfr:5d}  死亡 {d}  "
              f"吃道具 {f:3d}  拿斧头 {a}  {'过关!' if cl else ''}")
        nes.reset() if hasattr(nes, "reset") else None
    del nes

    xs = np.array(xs)
    print()
    print(f"局数 {len(xs)}  平均 {xs.mean():.1f} px  中位 {np.median(xs):.1f}  "
          f"最好 {xs.max():.1f}  最差 {xs.min():.1f}")
    print(f"过关 {nclear}/{len(xs)} 局   平均死亡 {np.mean(ds):.2f} 次/局   "
          f"平均吃道具 {np.mean(fs):.1f} 个/局   平均拿斧头 {np.mean(ax):.2f} 次/局")
    print(f"参考: 一屏宽 256px → 平均推进 {xs.mean()/256:.1f} 屏")


if __name__ == "__main__":
    main()
