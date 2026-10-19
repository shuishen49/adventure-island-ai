"""id 67 and id 198 kill the runner at ~5500 px on 1-4 -- are they the boss?

tools/diag_killer_id.py on runs/ppo_14d (12 episodes, average 5051 px,
best 5545): 9 of 12 deaths happen with **both id 67 and id 198 sitting
about 21 px to the right of the player**, one roughly 34 px above and one
roughly 5 px below -- a ~40 px tall thing built from two slots.  That is
exactly what a two-sprite boss looks like, and the user confirmed 1-4 is
cleared by killing a boss.

So, from the deepest honest state (saves/stages/stage_1-4_deep.state):
  1. show what they look like (zoomed ASCII + an annotated PNG);
  2. throw axes at them and watch whether the slots ever clear;
  3. diff the whole 2 KB of RAM for a byte that only ever goes DOWN --
     the boss HP, which is what a damage reward needs.

    python tools/probe_boss_pair.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,      # noqa: E402
                 NES_INPUT_A, NES_INPUT_B, NES_INPUT_LEFT)

try:
    from PIL import Image, ImageDraw
except ImportError:                                        # pragma: no cover
    Image = None

RAMP = " .:-=+*#%@"
N_RAM = 2048
TARGETS = (67, 198)


def ascii_art(frame, w=64, h=26, box=None):
    g = frame.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    if box is not None:
        x0, y0, x1, y1 = box
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(g.shape[1], int(x1)), min(g.shape[0], int(y1))
        g = g[y0:y1, x0:x1]
        if g.size == 0:
            return "(empty crop)"
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
        if oid in TARGETS:
            rgb = (255, 60, 255)
            label = f"BOSS?{oid}"
        x0, y0 = int(sx - 12) * scale, int(oy - 14) * scale
        x1, y1 = int(sx + 12) * scale, int(oy + 14) * scale
        d.rectangle([x0, y0, x1, y1], outline=rgb, width=2)
        d.text((x0, max(0, y0 - 12)),
               f"{label} dx{ox - px:+.0f} dy{oy - py:+.0f}", fill=rgb)
    d.rectangle([int(96 - 12) * scale, int(py - 14) * scale,
                 int(96 + 12) * scale, int(py + 14) * scale],
                outline=(60, 255, 60), width=2)
    return img


def slots_of(nes, ids):
    out = []
    for s in range(1, 16):
        v = int(nes[R.OBJ.id + R.OBJ.STRIDE * s])
        if v in ids:
            out.append((s, v))
    return out


def snap(nes):
    return np.frombuffer(bytes(nes[a] for a in range(N_RAM)),
                         dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_deep.state"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs",
                                                  "bosspair"))
    ap.add_argument("--frames", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3, start_state=args.state)
    env.reset(seed=0)
    nes = env._nes
    px = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    py = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    camx = px - 96.0

    print(f"loaded {args.state}")
    print(f"player=({px:.0f},{py:.0f}) DEATH={int(nes[R.DEATH])} "
          f"LIVES={int(nes[R.LIVES])} $007A={int(nes[R.STAGE_COMPLETE])}")
    hit = slots_of(nes, TARGETS)
    print(f"target slots present: {hit}")
    if hit:
        s0 = hit[0][0]
        ox = R.px(R.read_obj_world_x(nes, s0))
        oy = R.px(R.read_obj_world_y(nes, s0))
        frame0 = np.asarray(nes.step(1), dtype=np.uint8)
        print("\n--- zoom on the 67/198 pair (screen x = "
              f"{ox - camx:.0f}) ---")
        print(ascii_art(frame0, box=(ox - camx - 30, oy - 45,
                                     ox - camx + 30, oy + 45)))
        annotate(frame0, nes).save(os.path.join(args.out, "pair_1.png"))

    # ---- fight: stand off and throw, watch the slots -----------------
    print("\n=== fight: throw axes, watch slots 67/198, diff RAM ===")
    seen, log = [], []
    deaths = 0
    prev_death = int(nes[R.DEATH])
    for f in range(args.frames):
        nes[R.WEAPON] = max(1, int(nes[R.WEAPON]))
        nes[R.STAMINA] = 10
        # face right, throw constantly, hop now and then
        nes.controller = NES_INPUT_B if f % 12 < 3 else 0
        if f % 90 < 8:
            nes.controller |= NES_INPUT_A
        fr = np.asarray(nes.step(1), dtype=np.uint8)
        d = int(nes[R.DEATH])
        if d != 0 and prev_death == 0:
            deaths += 1
            print(f"  >> died at frame {f}")
        prev_death = d
        if f % 250 == 0:
            cur = slots_of(nes, TARGETS)
            log.append((f, cur))
            print(f"  f={f:5d} slots67/198={cur} "
                  f"$007A={int(nes[R.STAGE_COMPLETE])} "
                  f"DEATH={d}", flush=True)
            seen.append(snap(nes))
        if f == 1500:
            annotate(fr, nes).save(os.path.join(args.out, "pair_2.png"))
    annotate(np.asarray(nes.step(1), dtype=np.uint8), nes).save(
        os.path.join(args.out, "pair_3.png"))
    print(f"  deaths during the fight: {deaths}")

    if len(seen) > 3:
        arr = np.stack(seen)
        drops = []
        for a in range(N_RAM):
            v = arr[:, a].astype(int)
            dec = sum(1 for i in range(1, len(v)) if v[i] < v[i - 1])
            inc = sum(1 for i in range(1, len(v)) if v[i] > v[i - 1])
            if dec >= 2 and inc == 0 and v[0] > 0 and v[0] < 250:
                drops.append((dec, a, v[0], v[-1]))
        drops.sort(reverse=True)
        print(f"  bytes that only ever decreased: {len(drops)}")
        for dec, a, v0, v1 in drops[:20]:
            print(f"    ${a:04X}: {v0} -> {v1}  ({dec} drops)")


if __name__ == "__main__":
    main()
