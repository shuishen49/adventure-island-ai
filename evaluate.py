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


def run_episode(net, nes, max_frames=4500, greedy=True):
    nes.load(nes._save_state) if hasattr(nes, "_save_state") else None
    boot(nes)
    reset_cam(nes)
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

    for i in range(max_frames):
        if i % 4 == 0:
            with torch.no_grad():
                logits = net(obs_tensor(nes))[0]
            action = int(logits.argmax()) if greedy else int(
                torch.distributions.Categorical(logits=logits).sample())
        nes.controller = ACTIONS[action][0]
        push_img(nes, np.asarray(nes.step(), dtype=np.uint8))
        p = cam.progress(nes)
        if p > max_x:
            max_x = p
        lv = nes[R.LIVES]
        if lv < last_lives:
            deaths += 1
            last_lives = lv
        sc = nes[0x06A1]
        if sc != last_score:
            fruits += 1
            last_score = sc
        w = nes[R.WEAPON]
        if w > last_weapon:
            axes += 1
        last_weapon = w
        if lv == 0:
            break
    nes.controller = 0
    return max_x, deaths, fruits, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_final.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=4500)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"loaded {args.checkpoint}  ({ck.get('steps', '?')} steps)")

    nes = NES(ROM)
    xs, ds, fs, ax = [], [], [], []
    for e in range(args.episodes):
        x, d, f, a = run_episode(net, nes, args.max_frames)
        xs.append(x)
        ds.append(d)
        fs.append(f)
        ax.append(a)
        print(f"  ep{e+1:2d}: 最远 {x:7.1f} px   死亡 {d}  吃道具 {f}  "
              f"拿斧头 {a}")
        nes.reset() if hasattr(nes, "reset") else None
    del nes

    xs = np.array(xs)
    print()
    print(f"局数 {len(xs)}  平均 {xs.mean():.1f} px  中位 {np.median(xs):.1f}  "
          f"最好 {xs.max():.1f}  最差 {xs.min():.1f}")
    print(f"平均死亡 {np.mean(ds):.2f} 次/局   平均吃道具 {np.mean(fs):.1f} 个/局"
          f"   平均拿斧头 {np.mean(ax):.2f} 次/局")
    print(f"参考: 一屏宽 256px → 平均推进 {xs.mean()/256:.1f} 屏")


if __name__ == "__main__":
    main()
