"""Save the DEEPEST point the policy honestly reaches on 1-4, and shoot it.

No cheats: object 199 (fairy) breaks stage completion (1-1 and 1-2 also
run forever with it), and poking DEATH traps the death routine.  So this
plays with the plain policy, keeps re-saving the state every time the run
sets a new depth record, and at the end writes annotated PNGs plus an
ASCII dump of the deepest frame.  Those pictures are what the user looks
at to name the boss.

Then, from that same state, it throws axes at whatever is in front and
diffs the whole 2 KB of RAM looking for a byte that only ever goes DOWN:
that byte is the boss HP and is what a damage reward needs.

    python tools/deepest_14.py
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
from env import (AdventureIslandEnv, N_ACTIONS, OBS_DIM,   # noqa: E402
                 MAX_STEPS)
from train import ActorCritic                              # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:                                        # pragma: no cover
    Image = None

RAMP = " .:-=+*#%@"
N_RAM = 2048


def ascii_art(frame, w=76, h=30):
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


def annotate(frame, nes, scale=3):
    img = Image.fromarray(frame).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    camx = px - 96.0
    for s in range(1, 16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        sx = ox - camx
        if not (-30 < sx < 290):
            continue
        label, rgb = R.OBJ_KINDS.get(oid, ("?", (255, 255, 255)))
        x0, y0 = int(sx - 12) * scale, int(oy - 14) * scale
        x1, y1 = int(sx + 12) * scale, int(oy + 14) * scale
        d.rectangle([x0, y0, x1, y1], outline=rgb, width=2)
        d.text((x0, max(0, y0 - 12)),
               f"{label} id{oid} dx{ox - px:+.0f} dy{oy - py:+.0f}",
               fill=rgb)
    d.rectangle([int(96 - 12) * scale, int(py - 14) * scale,
                 int(96 + 12) * scale, int(py + 14) * scale],
                outline=(60, 255, 60), width=2)
    d.text((4, 4), "PLAYER = green", fill=(60, 255, 60))
    return img


def snap(nes):
    return np.frombuffer(bytes(nes[a] for a in range(N_RAM)),
                         dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs",
                                                  "deep"))
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_deep.state"))
    ap.add_argument("--fight-frames", type=int, default=3000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.state), exist_ok=True)

    if args.checkpoint is None:
        import glob
        c = sorted(glob.glob(os.path.join(_ROOT, "runs", "ppo_14d",
                                          "*.pt")))
        args.checkpoint = c[-1] if c else "runs/ppo_14c/ppo_final.pt"
    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    print(f"model={args.checkpoint}")

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    best = -1.0
    best_info = None
    for ep in range(args.episodes):
        o = env.reset(seed=1000 + ep)
        nes = env._nes
        n = 0
        while n < MAX_STEPS:
            with torch.no_grad():
                a = int(net({"ram": torch.from_numpy(o["ram"][None]),
                             "img": torch.from_numpy(o["img"][None])}
                            )[0].argmax(-1)[0])
            o, r, done, info = env.step(a)
            n += 1
            p = info.get("best_progress", 0.0)
            if p > best + 50:
                best = p
                with open(args.state, "wb") as fh:
                    fh.write(bytes(nes.save()))
                best_info = (ep, n, p)
            if done:
                break
        print(f"  ep{ep}: best {info.get('best_progress', 0):.0f} px "
              f"(record {best:.0f})", flush=True)
    print(f"\ndeepest honest run: {best:.0f} px  {best_info}")
    print(f"state -> {args.state}")

    # ---- describe + shoot the deepest frame --------------------------
    env2 = AdventureIslandEnv(start_with_axe=True, start_round=3,
                              stage_index=3, start_state=args.state)
    env2.reset(seed=0)
    nes = env2._nes
    px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    frame = np.asarray(nes.step(1), dtype=np.uint8)
    print(f"\nplayer=({px:.0f},{py:.0f}) $007A="
          f"{int(nes[R.STAGE_COMPLETE])} DEATH={int(nes[R.DEATH])} "
          f"LIVES={int(nes[R.LIVES])}")
    for s in range(16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        label = R.OBJ_KINDS.get(oid, ("?",))[0]
        print(f"  slot{s:2d} id{oid:3d} {label:8s} "
              f"dx{ox - px:+7.0f} dy{oy - py:+7.0f}")
    print(ascii_art(frame))
    annotate(frame, nes).save(os.path.join(args.out, "deep_1.png"))
    print(f"  -> {args.out}/deep_1.png")

    print("\n=== throw axes from here: hunt a byte that only drops ===")
    seen = []
    for f in range(args.fight_frames):
        nes[R.WEAPON] = max(1, int(nes[R.WEAPON]))
        nes.controller = 0x01 | (0x80 if f % 30 < 6 else 0) \
            | (0x40 if f % 10 < 2 else 0)
        fr = np.asarray(nes.step(1), dtype=np.uint8)
        if f % 40 == 0:
            seen.append(snap(nes))
        if f == 600:
            annotate(fr, nes).save(os.path.join(args.out, "deep_2.png"))
    if len(seen) > 3:
        arr = np.stack(seen)
        drops = []
        for a in range(N_RAM):
            v = arr[:, a].astype(int)
            dec = sum(1 for i in range(1, len(v)) if v[i] < v[i - 1])
            inc = sum(1 for i in range(1, len(v)) if v[i] > v[i - 1])
            if dec >= 2 and inc == 0 and v[0] > 0:
                drops.append((dec, a, v[0], v[-1]))
        drops.sort(reverse=True)
        print(f"  bytes that only ever decreased: {len(drops)}")
        for dec, a, v0, v1 in drops[:20]:
            print(f"    ${a:04X}: {v0} -> {v1}  ({dec} drops)")
    print(f"  $007A={int(nes[R.STAGE_COMPLETE])} "
          f"DEATH={int(nes[R.DEATH])} LIVES={int(nes[R.LIVES])}")
    annotate(np.asarray(nes.step(1), dtype=np.uint8), nes).save(
        os.path.join(args.out, "deep_3.png"))


if __name__ == "__main__":
    main()
