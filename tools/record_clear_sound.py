"""Re-record the clear run AND log every controller byte, so the exact same
run can be replayed in an audio-capable emulator (jsnes) to recover the
game's original soundtrack (cynes' APU is timing-only, it makes no sound).

Outputs (winning episode):
  outputs/adventure_island_clear_sound.mp4   annotated video, same style
  outputs/replay_inputs.bin                  1 byte per game frame (ctrl mask),
                                             episode phase only (post-boot)
  outputs/replay_meta.json                   boot/warmup offsets + timing
  outputs/replay_frames/f%06d.ppm            raw frame every --dump-every frames
  outputs/replay_progress.json               camera/lives sampled per dump

usage:
    python tools/record_clear_sound.py [ckpt] [--episodes 8]
"""

from __future__ import annotations

import json
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
import env as env_mod  # noqa: E402
from env import ACTIONS, OBS_DIM, AdventureIslandEnv, NES_INPUT_START  # noqa: E402
from train import ActorCritic  # noqa: E402
from visualize import annotate  # noqa: E402

OUT = os.path.join(_ROOT, "outputs")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_v10/ppo_final.pt"
N_EP = 8
FPS = 60.098804          # true NTSC rate -> audio and video timelines agree
SCALE = 4
DUMP_EVERY = 30          # raw frame dump cadence (video frames)
for i, a in enumerate(sys.argv):
    if a == "--episodes" and i + 1 < len(sys.argv):
        N_EP = int(sys.argv[i + 1])

MAX_FRAMES = int(FPS * 300)

# env._reset_internal_state boots with exactly this sequence before the
# save-state snapshot is taken; jsnes replays the same bytes from power-on.
BOOT_SEQ = [(300, 0x00), (60, int(NES_INPUT_START)), (60, 0x00)]
BOOT_FRAMES = sum(n for n, _ in BOOT_SEQ)


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
    if scale == 1:
        return frame
    return np.asarray(Image.fromarray(frame, mode="RGB").resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST),
        dtype=np.uint8)


def dump_ppm(arr: np.ndarray, path: str) -> None:
    h, w, _ = arr.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(arr.tobytes())


def play_one(net, env, inputs, path, dump_dir, progress_log):
    """Play one episode; append every controller byte to `inputs`.

    `inputs` is indexed by game frames AFTER the boot sequence (the caller
    seeds it with the logged warm-up + settle frame before calling us).
    """
    nes = env._nes
    tracker = R.SlotActivityTracker()
    cam = env._cam
    start_lives = env._start_lives

    writer = imageio.get_writer(path, fps=FPS, codec="libx264",
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
            ctrl = ACTIONS[action][0]
            inputs.append(ctrl)
            nes.controller = ctrl
            arr = np.asarray(nes.step(), dtype=np.uint8)
            nes.controller = 0
            env._push_frame(arr)
            env._last_frame = arr

            prog = cam.progress(nes)
            best = max(best, prog)
            extra = f"progress {prog:7.0f} px"
            if nes[R.STAGE_COMPLETE] != 0:
                extra += "   *** STAGE CLEAR ***"

            if frames % DUMP_EVERY == 0:
                dump_ppm(arr, os.path.join(dump_dir, f"f{frames:06d}.ppm"))
                progress_log.append({
                    "frame": frames,
                    "camera": int(nes[0x0593]),
                    "lives": int(nes[R.LIVES]),
                    "progress": round(float(prog), 1),
                })

            writer.append_data(upscale(
                annotate(arr, nes, ACTIONS[action][1], tracker, extra),
                SCALE))
            frames += 1

            if nes[R.STAGE_COMPLETE] != 0:
                win = True
                for _ in range(int(FPS) * 3):       # let the fanfare play
                    inputs.append(0x00)
                    nes.controller = 0
                    arr = np.asarray(nes.step(), dtype=np.uint8)
                    if frames % DUMP_EVERY == 0:
                        dump_ppm(arr, os.path.join(dump_dir, f"f{frames:06d}.ppm"))
                        progress_log.append({
                            "frame": frames,
                            "camera": int(nes[0x0593]),
                            "lives": int(nes[R.LIVES]),
                            "progress": round(float(prog), 1),
                        })
                    writer.append_data(upscale(
                        annotate(arr, nes, "-", tracker,
                                 f"progress {prog:7.0f} px"
                                 "   *** STAGE CLEAR ***"),
                        SCALE))
                    frames += 1
                break
            if nes[R.DEATH] != 0 or nes[R.LIVES] < start_lives:
                for _ in range(int(FPS)):           # show the death animation
                    inputs.append(0x00)
                    nes.controller = 0
                    arr = np.asarray(nes.step(), dtype=np.uint8)
                    if frames % DUMP_EVERY == 0:
                        dump_ppm(arr, os.path.join(dump_dir, f"f{frames:06d}.ppm"))
                        progress_log.append({
                            "frame": frames,
                            "camera": int(nes[0x0593]),
                            "lives": int(nes[R.LIVES]),
                            "progress": round(float(prog), 1),
                        })
                    writer.append_data(upscale(
                        annotate(arr, nes, "-", tracker,
                                 f"progress {prog:7.0f} px   *** DEAD ***"),
                        SCALE))
                    frames += 1
                break
    finally:
        writer.close()
    return {"win": win, "best": best, "frames": frames, "path": path}


def main():
    os.makedirs(OUT, exist_ok=True)
    net = load(CKPT)
    env = AdventureIslandEnv()

    # log the warm-up inputs that env.reset() plays internally
    warmup_log: list[int] = []

    def logged_warmup():
        n = int(env.rng.integers(0, env_mod.WARMUP_MAX + 1))
        for _ in range(n):
            c = int(env.rng.choice(
                (0x00, env_mod.NES_INPUT_RIGHT,
                 env_mod.NES_INPUT_RIGHT | env_mod.NES_INPUT_A)))
            warmup_log.append(c)
            env._nes.controller = c
            env._nes.step()
        env._nes.controller = 0

    env._warmup = logged_warmup

    for ep in range(N_EP):
        inputs: list[int] = []
        warmup_log.clear()
        env.reset()
        inputs.extend(warmup_log)
        inputs.append(0x00)          # the settle frame env.reset() steps

        # per-episode dump dir: no cleanup, failed episodes are simply kept
        dump_dir = os.path.join(OUT, f"replay_frames_ep{ep}")
        os.makedirs(dump_dir, exist_ok=True)
        progress_log: list[dict] = []

        tmp = os.path.join(OUT, f"_rec_sound_ep{ep}.mp4")
        r = play_one(net, env, inputs, tmp, dump_dir, progress_log)
        print(f"  ep{ep + 1}: {'CLEAR!' if r['win'] else 'died':<6} "
              f"{r['best']:7.0f} px   {r['frames']:5d} frames "
              f"({r['frames'] / FPS:.0f}s)", flush=True)

        if r["win"]:
            warmup_n = len(warmup_log)
            meta = {
                "checkpoint": CKPT,
                "fps_video": FPS,
                "fps_nes": 60.098804,
                "sample_rate": 44100,
                "boot_frames": BOOT_FRAMES,
                "boot_seq": BOOT_SEQ,
                "warmup_frames": warmup_n,
                "settle_frames": 1,
                "video_frames": r["frames"],
                # game-frame index (from power-on) of video frame 0
                "audio_start_frame": BOOT_FRAMES + warmup_n + 1,
                "dump_dir": dump_dir,
                "nes_palette_note": "raw dumps are 256x240 RGB",
            }
            with open(os.path.join(OUT, "replay_meta.json"), "w") as f:
                json.dump(meta, f, indent=1)
            with open(os.path.join(OUT, "replay_inputs.bin"), "wb") as f:
                f.write(bytes(inputs))
            with open(os.path.join(OUT, "replay_progress.json"), "w") as f:
                json.dump(progress_log, f, indent=1)

            final = os.path.join(OUT, "adventure_island_clear_sound.mp4")
            if os.path.exists(final):
                os.remove(final)
            os.replace(tmp, final)
            print(f"\ninputs: {len(inputs)} bytes "
                  f"(warmup {warmup_n} + settle 1 + episode "
                  f"{len(inputs) - warmup_n - 1})")
            print(f"video -> {final}  ({r['frames'] / FPS:.1f}s)")
            break

        if os.path.exists(tmp):
            os.remove(tmp)
    else:
        print("no clear this time; rerun for another batch of episodes")

    del env


if __name__ == "__main__":
    main()
