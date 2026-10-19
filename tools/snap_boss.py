"""Get to the end of 1-4, SCREENSHOT it, and probe how the boss takes damage.

The user watched the live window and said the runner reaches "the boss",
and confirmed 1-4 is cleared by KILLING A BOSS (not by walking to a flag).
Earlier measurement: with real invincibility the runner scrolls out to
~25000 px, the camera then STOPS, the screen fills with id 56 spiders and
an id 57 sits nearby -- that has to be the arena.

What this script does:
  1. drive there with the trained policy + the fairy (id 199) cheat;
  2. save annotated PNGs (every object boxed and labelled) plus an ASCII
     dump, because I cannot look at PNGs myself but the user can;
  3. write the arena to saves/stages/stage_1-4_boss.state so every later
     experiment can start there instantly;
  4. once there, throw axes at whatever is on screen and diff the WHOLE
     2 KB of RAM to find bytes that only ever DECREASE -- that is the
     boss HP byte (the thing the reward function needs).

    python tools/snap_boss.py
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

RAMP = " .:-=+*#%@"
N_RAM = 2048


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


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))


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
               f"{label} id{oid} dx{ox - px:+.0f} dy{oy - py:+.0f}",
               fill=rgb)
    d.rectangle([int(96 - 10) * scale, int(py - 12) * scale,
                 int(96 + 10) * scale, int(py + 12) * scale],
                outline=(60, 255, 60), width=2)
    d.text((4, 4), "PLAYER = green box", fill=(60, 255, 60))
    return img


def force_fairy(nes) -> None:
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


def snap(nes) -> np.ndarray:
    return np.frombuffer(bytes(nes[a] for a in range(N_RAM)),
                         dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/ppo_14c/ppo_final.pt")
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "boss"))
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_boss.state"))
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--fight-frames", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.state), exist_ok=True)

    ck = torch.load(args.checkpoint, map_location="cpu",
                    weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    cam = env._cam

    print("=== phase 1: drive to the arena (fairy + trained policy) ===")
    best, stall, shots = 0.0, 0, 0
    arena_at = None
    for step in range(args.max_steps):
        force_fairy(nes)
        nes[R.STAMINA] = 10
        nes[R.LIVES] = 9
        o = env._obs()
        with torch.no_grad():
            a = int(net({"img": torch.as_tensor(o["img"][None]),
                         "ram": torch.as_tensor(o["ram"][None],
                                                dtype=torch.float32)}
                        )[0].argmax())
        nes.controller = ACTIONS[a][0]
        arr = None
        for _ in range(env.frame_skip):
            arr = np.asarray(nes.step(), dtype=np.uint8)
        nes.controller = 0
        env._push_frame(arr)
        p = cam.progress(nes)
        if p > best + 0.5:
            best, stall = p, 0
        else:
            stall += 1
        if step % 400 == 0:
            print(f"  step={step:6d} prog={p:7.0f} best={best:7.0f} "
                  f"cam={int(nes[R.CAMERA]):3d} $007A="
                  f"{int(nes[R.STAGE_COMPLETE])}", flush=True)
        if stall > 400 and arena_at is None:
            arena_at = step
            print(f"  >> camera stopped at {best:.0f} px (step {step})",
                  flush=True)
        # keep shooting while the camera is stopped
        if arena_at is not None and shots < args.shots \
                and (step - arena_at) % 25 == 0 and step != arena_at:
            path = os.path.join(args.out, f"boss_{shots + 1}.png")
            annotate(arr, nes).save(path)
            print(f"     shot -> {path}  (step {step})", flush=True)
            shots += 1
        if shots >= args.shots:
            break

    if arena_at is None:
        print("  !! never reached an arena; best was "
              f"{best:.0f} px", flush=True)
        return

    with open(args.state, "wb") as fh:
        fh.write(bytes(nes.save()))
    print(f"  arena state -> {args.state}", flush=True)

    print("\n--- objects in the arena ---")
    px, py = player_xy(nes)
    for s in range(16):
        oid = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if oid == 0:
            continue
        ox = R.px(R.read_obj_world_x(nes, s))
        oy = R.px(R.read_obj_world_y(nes, s))
        label = R.OBJ_KINDS.get(oid, ("?",))[0]
        print(f"  slot{s:2d} id{oid:3d} {label:8s} "
              f"dx{ox - px:+7.0f} dy{oy - py:+7.0f}")
    print(ascii_art(arr))

    print("\n=== phase 2: throw axes, hunt for a byte that only drops ===")
    base = snap(nes)
    seen: dict[int, list[int]] = {}
    for f in range(args.fight_frames):
        force_fairy(nes)
        nes[R.STAMINA] = 10
        nes[R.LIVES] = 9
        nes[R.WEAPON] = max(1, int(nes[R.WEAPON]))
        # hold right, jump, and throw constantly
        nes.controller = 0x01 | (0x80 if f % 30 < 6 else 0) \
            | (0x40 if f % 8 < 2 else 0)
        nes.step(1)
        if f % 40 == 0:
            cur = snap(nes)
            for a2 in range(N_RAM):
                seen.setdefault(a2, []).append(int(cur[a2]))
    print(f"  sampled {len(next(iter(seen.values())))} times")
    drops = []
    for a2, vals in seen.items():
        dec = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i - 1])
        inc = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i - 1])
        if dec >= 3 and inc == 0 and vals[0] > 0:
            drops.append((dec, a2, vals[0], vals[-1]))
    drops.sort(reverse=True)
    print(f"  bytes that only ever decreased: {len(drops)}")
    for dec, a2, v0, v1 in drops[:20]:
        print(f"    ${a2:04X}: {v0} -> {v1}  ({dec} drops, 0 rises)")
    print(f"  $007A (stage clear) = {int(nes[R.STAGE_COMPLETE])}")


if __name__ == "__main__":
    main()
