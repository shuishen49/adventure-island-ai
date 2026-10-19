"""Start from a dumped save state, play with the policy, screenshot it.

Unlike snap_boss.py this starts from ANY state (usually one captured by
tools/reach_14_late.py at a chosen progress) and does NOT use any cheat:
whatever the policy does from here is what it would really do.

    python tools/snap_at.py --state saves/stages/stage_1-4_5400.state
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
from env import (AdventureIslandEnv, ACTIONS, N_ACTIONS,    # noqa: E402
                 OBS_DIM)
from train import ActorCritic                              # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:                                        # pragma: no cover
    Image = None


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, 0)),
            R.px(R.read_obj_world_y(nes, 0)))


def annotate(frame, nes, scale=3):
    img = Image.fromarray(frame).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    px, py = player_xy(nes)
    cam = px - 96.0
    for s in range(1, 16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        label, rgb = R.OBJ_KINDS.get(oid, ("?", (255, 255, 255)))
        x0, y0 = int(ox - cam - 10) * scale, int(oy - 12) * scale
        x1, y1 = int(ox - cam + 10) * scale, int(oy + 12) * scale
        d.rectangle([x0, y0, x1, y1], outline=rgb, width=2)
        d.text((x0, max(0, y0 - 12)),
               f"s{s} {label} id{oid} dx{ox - px:+.0f} dy{oy - py:+.0f}",
               fill=rgb)
    d.rectangle([int(96 - 10) * scale, int(py - 12) * scale,
                 int(96 + 10) * scale, int(py + 12) * scale],
                outline=(60, 255, 60), width=2)
    d.text((4, 4), "PLAYER = green box", fill=(60, 255, 60))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--checkpoint",
                    default=os.path.join(_ROOT, "runs", "ppo_14d",
                                         "ppo_27770880.pt"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "at"))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--shot-every", type=int, default=25)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"model={args.checkpoint} state={args.state}", flush=True)

    n = 0
    for ep in range(args.episodes):
        env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                                 stage_index=3, start_state=args.state)
        env.reset(seed=ep)
        nes = env._nes
        for step in range(args.max_steps):
            o = env._obs()
            with torch.no_grad():
                a = int(net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)}
                            )[0].argmax())
            o, r, done, info = env.step(a)
            if step % args.shot_every == 0 or done:
                n += 1
                path = os.path.join(args.out, f"at_{n}.png")
                g = (o["img"][:, :, -1] * 255).astype(np.uint8)
                frame3 = np.repeat(g[:, :, None], 3, axis=2)
                annotate(frame3, nes).save(path)
                tag = "DEATH" if done else f"step{step}"
                print(f"ep{ep} {tag}: {path}  "
                      f"alive={int(nes[R.LIVES])} "
                      f"slots={[int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) for s in range(1, 16) if int(nes[R.OBJ.id + R.OBJ.STRIDE * s])]}",
                      flush=True)
            if done:
                break
    print(f"{n} shots -> {args.out}")


if __name__ == "__main__":
    main()
