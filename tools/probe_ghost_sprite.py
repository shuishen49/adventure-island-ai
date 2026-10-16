"""Probe: is the "visible snail with no slot" a live enemy or a stale sprite?

Every 10 frames: detect snail-pink pixel blobs (excluding the player box),
list enemy slots, and match blobs to slots.  If an unmatched blob MOVES
over time the game is updating it -> there must be slot data we misread.
If it never moves and the player can overlap it safely -> display leftover.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from cynes import NES

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, reset_cam
from train import ActorCritic
from visualize import boot, objects

# snail body/shell magenta-pink (tight, to exclude lighter vine flowers)
def pink_mask(frame):
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)
    return (r > 170) & (r - g > 80) & (b > 60) & (b < 170) & (g < 120)


def blobs(mask, min_cells=3):
    """Coarse 4x4-cell connected components."""
    m = mask[::4, ::4]
    seen = np.zeros_like(m, dtype=bool)
    out = []
    H, W = m.shape
    for y in range(H):
        for x in range(W):
            if m[y, x] and not seen[y, x]:
                stack = [(y, x)]
                seen[y, x] = True
                cells = []
                while stack:
                    cy, cx = stack.pop()
                    cells.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < H and 0 <= nx < W
                                    and m[ny, nx] and not seen[ny, nx]):
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                if len(cells) >= min_cells:
                    ys = [c[0] for c in cells]
                    xs = [c[1] for c in cells]
                    out.append((float(np.mean(xs)) * 4,
                                float(np.mean(ys)) * 4, len(cells)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_00092160.pt")
    ap.add_argument("--max-frames", type=int, default=5000)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    boot(nes)
    reset_cam(nes)

    action = 0
    unmatched_hist = {}     # blob key -> last position, to see if it moves
    for i in range(args.max_frames):
        if i % 4 == 0:
            with torch.no_grad():
                action = int(net(obs_tensor(nes))[0].argmax())
        nes.controller = ACTIONS[action][0]
        frame = np.asarray(nes.step(), dtype=np.uint8)
        if i % 10:
            continue

        objs = objects(nes)
        enemies = [(s, x, y, v) for s, oid, x, y, v in objs
                   if oid in R.ENEMY_IDS]
        p_box = [(o[2], o[3]) for o in objs if o[0] == 0]
        mask = pink_mask(frame)
        if p_box:
            px_, py_ = p_box[0]
            mask[max(0, int(py_) - 30):int(py_) + 8,
                 max(0, int(px_) - 14):int(px_) + 14] = False
        bl = blobs(mask)
        unmatched = []
        for bx, by, n in bl:
            near = [e for e in enemies
                    if abs(e[1] - bx) < 24 and abs(e[2] - by) < 24]
            if not near:
                unmatched.append((bx, by, n))
        if unmatched or i % 300 == 0:
            print(f"f{i:5d} P={nes[R.OBJ.x_lo]:3.0f} "
                  f"slots={[(s, f'{x:.0f}', f'{y:.0f}') for s, x, y, v in enemies]} "
                  f"blobs={[(f'{x:.0f}', f'{y:.0f}') for x, y, n in bl]} "
                  f"unmatched={[(f'{x:.0f}', f'{y:.0f}') for x, y, n in unmatched]} "
                  f"lives={nes[R.LIVES]}")

    nes.controller = 0
    del nes


if __name__ == "__main__":
    main()
