"""Record the agent's best run as an annotated MP4.

Every game frame is drawn with the same overlay the live window uses:
coloured boxes for the player / enemies (id14 snail, id9 bird) / fruit /
item eggs / campfires (id2) / id13, the distance to each object, a line
from the player to anything within 140 px, plus a HUD with lives, stamina,
axe, area/round and the pixel progress.

Several episodes are played; the one that CLEARS the stage wins, otherwise
the one that got the farthest.  Only that video is kept.

usage:
    python tools/record_clear.py [ckpt] [--episodes 6] [--fps 60]
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

import imageio.v2 as imageio  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic  # noqa: E402
from visualize import annotate  # noqa: E402

OUT = os.path.join(_ROOT, "outputs")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_v10/ppo_final.pt"
N_EP = 6
FPS = 60
# 2026-10-19: which stage to record.  Without these the script always
# booted 1-1, so the first real 1-4 clear (runs/ppo_14n) could not be
# captured: the agent has to see the one-hot bit of the stage it plays.
START_STATE = None      # e.g. saves/stages/stage_1-4.state
START_ROUND = 0         # 3 = 第四小关 (1-4)
STAGE_IDX = 3           # one-hot bit: 3 = 1-4, 4 = 2-1
FINAL_OUT = os.path.join(OUT, "adventure_island_clear.mp4")
SCALE = 4
for i, a in enumerate(sys.argv):
    if a == "--episodes" and i + 1 < len(sys.argv):
        N_EP = int(sys.argv[i + 1])
    if a == "--fps" and i + 1 < len(sys.argv):
        FPS = int(sys.argv[i + 1])
    if a == "--scale" and i + 1 < len(sys.argv):
        SCALE = int(sys.argv[i + 1])
    if a == "--start-state" and i + 1 < len(sys.argv):
        START_STATE = sys.argv[i + 1]
    if a == "--start-round" and i + 1 < len(sys.argv):
        START_ROUND = int(sys.argv[i + 1])
    if a == "--stage-idx" and i + 1 < len(sys.argv):
        STAGE_IDX = int(sys.argv[i + 1])
    if a == "--out" and i + 1 < len(sys.argv):
        FINAL_OUT = sys.argv[i + 1]

# ~5 minutes of game time; a clear of stage 1-1 fits well inside this.
MAX_FRAMES = FPS * 300


def load(path):
    net = ActorCritic(OBS_DIM, len(ACTIONS))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    own = net.state_dict()
    keep = {k: v for k, v in ck["model"].items()
            if k in own and own[k].shape == v.shape}
    net.load_state_dict(keep, strict=False)
    net.eval()
    print(f"loaded {path} ({ck.get('steps', 0)} env steps, "
          f"{len(keep)}/{len(ck['model'])} tensors)")
    return net


def upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    """256x240 -> scale * that, NEAREST so the pixels stay sharp."""
    if scale == 1:
        return frame
    return np.asarray(Image.fromarray(frame, mode="RGB").resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST),
        dtype=np.uint8)


def play_one(net, env, path, fps, scale):
    """Play a single episode, streaming annotated frames into `path`."""
    obs = env.reset()
    nes = env._nes
    tracker = R.SlotActivityTracker()
    cam = env._cam
    start_lives = env._start_lives

    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                quality=None, pixelformat="yuv420p",
                                output_params=["-crf", "16",
                                               "-preset", "medium"])
    action = 0
    win = False
    frames = 0
    best = 0.0
    try:
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
            env._last_frame = arr

            prog = cam.progress(nes)
            best = max(best, prog)
            extra = f"progress {prog:7.0f} px"
            if nes[R.STAGE_COMPLETE] != 0:
                extra += "   *** STAGE CLEAR ***"
            writer.append_data(upscale(
                annotate(arr, nes, ACTIONS[action][1], tracker, extra),
                scale))
            frames += 1

            if nes[R.STAGE_COMPLETE] != 0:
                win = True
                # let the fanfare play out a bit
                for _ in range(fps * 3):
                    nes.controller = 0
                    arr = np.asarray(nes.step(), dtype=np.uint8)
                    writer.append_data(upscale(
                        annotate(arr, nes, "-", tracker,
                                 f"progress {prog:7.0f} px"
                                 "   *** STAGE CLEAR ***"), scale))
                    frames += 1
                break
            if nes[R.DEATH] != 0 or nes[R.LIVES] < start_lives:
                for _ in range(fps):      # show the death animation
                    nes.controller = 0
                    arr = np.asarray(nes.step(), dtype=np.uint8)
                    writer.append_data(upscale(
                        annotate(arr, nes, "-", tracker,
                                 f"progress {prog:7.0f} px   *** DEAD ***"),
                        scale))
                    frames += 1
                break
    finally:
        writer.close()
    return {"win": win, "best": best, "frames": frames, "path": path}


def main():
    os.makedirs(OUT, exist_ok=True)
    net = load(CKPT)
    env = AdventureIslandEnv(start_round=START_ROUND,
                             stage_index=STAGE_IDX,
                             start_state=START_STATE)
    results = []
    try:
        for ep in range(N_EP):
            tmp = os.path.join(OUT, f"_rec_ep{ep}.mp4")
            r = play_one(net, env, tmp, FPS, SCALE)
            results.append(r)
            print(f"  ep{ep + 1}: {'CLEAR!' if r['win'] else 'died':<6} "
                  f"{r['best']:7.0f} px   {r['frames']:5d} frames "
                  f"({r['frames'] / FPS:.0f}s)", flush=True)
            if r["win"]:
                break
    finally:
        env.close()

    best = max(results, key=lambda r: (r["win"], r["best"]))
    final = FINAL_OUT
    os.makedirs(os.path.dirname(os.path.abspath(final)), exist_ok=True)
    if os.path.exists(final):
        os.remove(final)
    os.replace(best["path"], final)
    for r in results:
        if r is not best and os.path.exists(r["path"]):
            os.remove(r["path"])

    print()
    print(f"winner: {os.path.basename(best['path'])} -> {final}")
    print(f"  cleared stage : {'YES' if best['win'] else 'no'}")
    print(f"  farthest      : {best['best']:.0f} px")
    print(f"  length        : {best['frames'] / FPS:.1f} s "
          f"({best['frames']} frames @ {FPS} fps)")
    print(f"  size          : {os.path.getsize(final) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
