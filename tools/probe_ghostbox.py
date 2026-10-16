"""Find which object slot produces a "ghost" box that never leaves the screen.

Runs the trained agent for a while and, for every object slot, reports how
often it is on screen and how much its computed screen position varies.
A slot with a high on-screen count and a near-zero position std is the
ghost: it is drawn fixed on screen even though no sprite is there.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, ".")

from cynes import NES

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, push_img, reset_cam
from train import ActorCritic
from visualize import boot

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo/ppo_00081920.pt"
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
net.load_state_dict(ck["model"])
net.eval()
print(f"agent: {CKPT} ({ck.get('steps', '?')} steps)")

nes = NES(ROM)
boot(nes)
reset_cam(nes)

stats = defaultdict(lambda: {"ids": set(), "n": 0, "on": 0,
                             "xs": [], "ys": []})
action = 0
for i in range(FRAMES):
    if i % 4 == 0:
        with torch.no_grad():
            idx = int(net(obs_tensor(nes))[0].argmax())
        action = ACTIONS[idx][0]
    nes.controller = action
    push_img(nes, np.asarray(nes.step(), dtype=np.uint8))

    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + s]
        if oid == 0:
            continue
        x = R.px(R.read_obj_world_x(nes, s))
        y = R.px(R.read_obj_world_y(nes, s))
        st = stats[s]
        st["ids"].add(oid)
        st["n"] += 1
        st["xs"].append(x)
        st["ys"].append(y)
        if 0 <= x < 256 and 0 <= y < 240:
            st["on"] += 1

print(f"\n{'slot':>4} {'ids':>12} {'frames':>7} {'onscreen':>8} "
      f"{'x range':>16} {'x std':>7} {'y range':>16} {'y std':>7}")
for s in sorted(stats):
    st = stats[s]
    xs = np.array(st["xs"])
    ys = np.array(st["ys"])
    tag = "  <== CAMERA SLOT" if s == R.OBJ.CAMERA_SLOT else ""
    print(f"{s:>4} {str(sorted(st['ids'])):>12} {st['n']:>7} {st['on']:>8} "
          f"{f'[{xs.min():7.1f},{xs.max():7.1f}]':>16} {xs.std():>7.2f} "
          f"{f'[{ys.min():7.1f},{ys.max():7.1f}]':>16} {ys.std():>7.2f}{tag}")

del nes
