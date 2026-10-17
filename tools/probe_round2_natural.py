"""Capture the natural Round 1-2 start state.

Plays stage 1-1 with the trained greedy policy until STAGE_COMPLETE, then
steps through the tally -> banner -> gameplay transition frame by frame,
logging the round register, the player slot and the player position, and
dumps screenshots.  The goal is to see exactly when "round 2 gameplay" has
begun so an equivalent save state can be taken (and compared against the
RAM-warp shortcut: set $0038 = 1 during stage init).

usage: python tools/probe_round2_natural.py [ckpt]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic  # noqa: E402

OUT = os.path.join(_ROOT, "outputs", "probe_round2")
os.makedirs(OUT, exist_ok=True)
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _ROOT, "runs", "ppo_v10", "ppo_final.pt")

MAX_FRAMES = 60 * 300          # ~5 min of game time, a clear fits inside


def load(path):
    net = ActorCritic(OBS_DIM, len(ACTIONS))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    own = net.state_dict()
    keep = {k: v for k, v in ck["model"].items()
            if k in own and own[k].shape == v.shape}
    net.load_state_dict(keep, strict=False)
    net.eval()
    print(f"loaded {path} ({ck.get('steps', 0)} env steps)")
    return net


def shot(nes, tag):
    arr = np.asarray(nes.step(), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").resize((512, 480), Image.NEAREST).save(
        os.path.join(OUT, f"{tag}.png"))
    return arr


def row(nes, i, tag=""):
    pid = nes[R.OBJ.id + R.PLAYER_SLOT]
    x = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
    print(f"  f{i:6d} {tag:<12} area={nes[R.AREA]} round={nes[R.ROUND]} "
          f"ckpt={nes[R.CHECKPOINT]} lives={nes[R.LIVES]} "
          f"clear={nes[R.STAGE_COMPLETE]} death={nes[R.DEATH]} "
          f"pid={pid} px=({x:6.1f},{y:6.1f})", flush=True)


def main():
    net = load(CKPT)
    env = AdventureIslandEnv()
    nes = env._nes
    try:
        obs = env.reset()
        action = 0
        cleared_at = None
        for f in range(MAX_FRAMES):
            if f % env.frame_skip == 0:
                o = env._obs()
                with torch.no_grad():
                    logits, _ = net({
                        "img": torch.as_tensor(o["img"][None]),
                        "ram": torch.as_tensor(o["ram"][None],
                                               dtype=torch.float32)})
                    action = int(logits.argmax())
            nes.controller = ACTIONS[action][0]
            arr = np.asarray(nes.step(), dtype=np.uint8)
            nes.controller = 0
            env._push_frame(arr)
            if nes[R.STAGE_COMPLETE] != 0:
                cleared_at = f
                print(f"STAGE CLEAR at frame {f} ({f / 60:.1f}s)")
                break
            if nes[R.DEATH] != 0:
                print(f"DIED at frame {f} ({f / 60:.1f}s) - aborting")
                return
        if cleared_at is None:
            print("never cleared stage 1-1 in time - aborting")
            return

        # --- transition: tally -> banner -> round 2 gameplay ------------
        print("\n=== transition frames (no input) ===")
        prev_round = nes[R.ROUND]
        round_changed_at = None
        for i in range(60 * 25):                     # up to 25 s
            nes.controller = 0
            nes.step()
            rnd = nes[R.ROUND]
            if rnd != prev_round:
                prev_round = rnd
                round_changed_at = i
                print(f"  --> ROUND {rnd} at +{i}f")
            if i % 30 == 0 or (round_changed_at is not None
                               and i < round_changed_at + 240
                               and i % 10 == 0):
                row(nes, i, "trans")
            # gameplay of the new round: player slot alive and standing
            if (rnd == 1 and nes[R.OBJ.id + R.PLAYER_SLOT] == R.PLAYER_ID
                    and i > (round_changed_at or 0) + 120):
                break
        row(nes, i, "candidate")
        shot(nes, "natural_round2_start")

        # --- control test on a COPY: load, hold right, check movement ---
        state = nes.save()
        x0 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        nes.controller = 0x01
        for _ in range(60):
            nes.step()
        nes.controller = 0
        x1 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
        print(f"\ncontrol test: x {x0:.1f} -> {x1:.1f} "
              f"(moved {x1 - x0:.1f}px)")
        shot(nes, "natural_round2_after_right")
        with open(os.path.join(OUT, "natural_round2.state"), "wb") as fh:
            fh.write(state)
        print("state dumped to", os.path.join(OUT, "natural_round2.state"),
              f"({len(state)} bytes)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
