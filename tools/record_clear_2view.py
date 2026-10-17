"""Record a stage clear TWICE: the annotated player view and the raw agent view.

Two videos are written from the very same game frames:

  *_player.mp4    256x240 -> 1024x960, NEAREST, with the live-window overlay
                  (object boxes, ids, distances) and a HUD.  Also gets the
                  real NES soundtrack (see the jsnes step below).
  *_agent.mp4     exactly what the policy's CNN sees: 80x86 grey, every 3rd
                  pixel sampled, upscaled 8x with NEAREST -> a mosaic.
                  No overlay, no HUD: this is the agent's eyeball.

Because cynes' APU makes no sound, the controller bytes of the winning run
are logged from POWER-ON (title screen, START, the $0038 warp, the warm-up
and the whole episode) and replayed in jsnes, which does emulate the APU:

    python tools/record_clear_2view.py runs/ppo_12/ppo_final.pt \
        --start-round 1 --stage-idx 0 --tag 1-2
    node tools/jsnes_replay2.mjs roms/<rom>.nes \
        outputs/v1-2_inputs.bin outputs/v1-2_meta.json outputs/v1-2_audio
    python tools/mux_video.py outputs/v1-2_player.mp4 \
        outputs/v1-2_audio/game_audio.wav outputs/v1-2_player_sound.mp4

usage:
    python tools/record_clear_2view.py [ckpt] [--episodes 6] [--tag 1-2]
"""

from __future__ import annotations

import argparse
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
from cynes import NES  # noqa: E402
from env import (ACTIONS, IMG_H, IMG_W, OBS_DIM, ROM, WARMUP_MAX,  # noqa: E402
                 AdventureIslandEnv, NES_INPUT_A, NES_INPUT_RIGHT,
                 NES_INPUT_START, to_grey)
from train import ActorCritic  # noqa: E402
from visualize import annotate  # noqa: E402

OUT = os.path.join(_ROOT, "outputs")
FPS = 60.098804          # true NTSC rate -> audio and video timelines agree
DUMP_EVERY = 30          # raw-frame dump cadence used by the sync check


# ----------------------------------------------------------------------
def load_net(path):
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


def agent_frame(arr: np.ndarray, scale: int, pad: int = 4) -> np.ndarray:
    """The observation the CNN gets: 80x86 grey, blown up with NEAREST."""
    grey = to_grey(arr)
    img = Image.fromarray(grey, mode="L").resize(
        (IMG_W * scale, IMG_H * scale), Image.NEAREST).convert("RGB")
    if pad:
        img = Image.fromarray(np.asarray(
            Image.new("RGB", (IMG_W * scale + 2 * pad, IMG_H * scale + 2 * pad),
                      (24, 24, 24))))
        img.paste(Image.fromarray(grey, mode="L").resize(
            (IMG_W * scale, IMG_H * scale), Image.NEAREST).convert("RGB"),
            (pad, pad))
    return np.asarray(img, dtype=np.uint8)


# ----------------------------------------------------------------------
# boot, byte for byte (so jsnes can replay it from power-on)
# ----------------------------------------------------------------------
def control_live(nes, frames: int = 30) -> bool:
    """Same probe env._control_live() uses.  It rewinds, so it costs jsnes
    nothing -- we only need it to know when to stop burning frames."""
    st = np.frombuffer(nes.save(), dtype=np.uint8).copy()
    x0 = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes.controller = 0
    for _ in range(frames):
        nes.step()
    x_idle = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes.load(st)
    nes.controller = NES_INPUT_RIGHT
    for _ in range(frames):
        nes.step()
    nes.controller = 0
    x_right = R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT))
    nes.load(st)
    return abs(x_idle - x0) < 2.0 and (x_right - x0) > 5.0


def boot_logged(round_value: int):
    """Power-on -> playable stage `round_value`, logging every controller byte.

    Returns (nes, inputs, poke_range).  The NES object is kept: the caller
    dumps its state to build the env from, so no boot happens twice.
    """
    nes = NES(ROM)
    inputs: list[int] = []

    def run(n: int, mask: int, poke: bool = False):
        for _ in range(n):
            if poke:
                nes[R.ROUND] = round_value
            nes.controller = mask
            nes.step()
            inputs.append(mask)
        nes.controller = 0

    run(300, 0x00)                      # title screen
    run(90, NES_INPUT_START, poke=True)  # START with $0038 hammered
    poke_range = [300, 300 + 90]

    # env._boot_to_round(): probe (rewound) then burn 10 idle frames a time
    for _ in range(60):
        if control_live(nes):
            break
        run(10, 0x00)
    # ...then wait out the ~180 black frames of the stage fade-in
    for _ in range(400):
        if float(np.asarray(nes.step()).mean()) > 30.0:
            inputs.append(0x00)
            break
        inputs.append(0x00)
    return nes, inputs, poke_range


# ----------------------------------------------------------------------
def play_one(net, env, base_inputs, player_path, agent_path,
             scale, agent_scale, greedy, seed, cam_state):
    nes = env._nes
    tracker = R.SlotActivityTracker()
    cam = env._cam
    start_lives = env._start_lives
    inputs = list(base_inputs)

    kw = dict(fps=FPS, codec="libx264", quality=None,
              pixelformat="yuv420p",
              output_params=["-crf", "16", "-preset", "medium"])
    w_player = imageio.get_writer(player_path, **kw)
    w_agent = imageio.get_writer(agent_path, **kw)

    action = 0
    win = False
    frames = 0
    best = 0.0
    dumps: list[dict] = []
    try:
        for f in range(int(FPS * 300)):
            if f % env.frame_skip == 0:
                o = env._obs()
                with torch.no_grad():
                    logits, _ = net({
                        "img": torch.as_tensor(o["img"][None]),
                        "ram": torch.as_tensor(o["ram"][None],
                                               dtype=torch.float32)})
                if greedy:
                    action = int(logits.argmax())
                else:
                    action = int(torch.distributions.Categorical(
                        logits=logits[0]).sample())
            ctrl = ACTIONS[action][0]
            inputs.append(ctrl)
            nes.controller = ctrl
            arr = np.asarray(nes.step(), dtype=np.uint8)
            nes.controller = 0
            # env.step() runs frame_skip NES frames and pushes ONLY the last
            # one, so the policy is trained on a stack whose channels are
            # `frame_skip` frames apart.  Pushing every frame here feeds a
            # 1-frame gap -- the same silent train/test mismatch that made
            # evaluate.py report a bogus "230 px" (2026-10-17).  It showed up
            # here as a scripted 824 px while eval_native saw 6511 px on the
            # very same seed.
            if f % env.frame_skip == env.frame_skip - 1:
                env._push_frame(arr)
            env._last_frame = arr

            prog = cam.progress(nes)
            best = max(best, prog)
            cleared = nes[R.STAGE_COMPLETE] != 0
            extra = f"progress {prog:7.0f} px"
            if cleared:
                extra += "   *** STAGE CLEAR ***"

            if frames % DUMP_EVERY == 0:
                dumps.append({"frame": frames,
                              "camera": int(nes[0x0593]),
                              "lives": int(nes[R.LIVES]),
                              "progress": round(float(prog), 1)})

            w_player.append_data(upscale(
                annotate(arr, nes, ACTIONS[action][1], tracker, extra), scale))
            w_agent.append_data(agent_frame(arr, agent_scale))
            frames += 1

            if cleared:
                win = True
                tail = int(FPS * 3)          # let the fanfare play out
            elif nes[R.DEATH] != 0 or nes[R.LIVES] < start_lives:
                tail = int(FPS)              # show the death animation
            else:
                continue
            for _ in range(tail):
                inputs.append(0x00)
                nes.controller = 0
                arr = np.asarray(nes.step(), dtype=np.uint8)
                w_player.append_data(upscale(
                    annotate(arr, nes, "-", tracker,
                             f"progress {prog:7.0f} px   "
                             + ("*** STAGE CLEAR ***" if win else "*** DEAD ***")),
                    scale))
                w_agent.append_data(agent_frame(arr, agent_scale))
                frames += 1
            break
    finally:
        w_player.close()
        w_agent.close()
    return {"win": win, "best": best, "frames": frames,
            "player": player_path, "agent": agent_path,
            "inputs": inputs, "dumps": dumps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?",
                    default="runs/ppo_12/ppo_final.pt")
    ap.add_argument("--start-round", type=int, default=1,
                    help="0-based $0038: 1 = stage 1-2")
    ap.add_argument("--stage-idx", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--tag", default="1-2")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--agent-scale", type=int, default=8)
    ap.add_argument("--no-greedy-first", action="store_true")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax in EVERY episode (default: only ep1).  Use "
                         "this when you know a certain seed clears -- see "
                         "outputs/find_clear_seed.log -- because sampling "
                         "throws that reproducibility away.")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    net = load_net(args.checkpoint)

    # ---- boot from power-on, logging every byte ----------------------
    nes, boot_inputs, poke_range = boot_logged(args.start_round)
    print(f"boot: {len(boot_inputs)} frames "
          f"(round poke every frame {poke_range[0]}..{poke_range[1] - 1})",
          flush=True)
    state_path = os.path.join(OUT, f"_boot_state_{args.tag}.state")
    with open(state_path, "wb") as fh:
        fh.write(nes.save())
    del nes

    env = AdventureIslandEnv(start_state=state_path, start_with_axe=True,
                             stage_index=args.stage_idx,
                             clear_ends_episode=True)

    warmup_log: list[int] = []

    def logged_warmup():
        n = int(env.rng.integers(0, WARMUP_MAX + 1))
        for _ in range(n):
            c = int(env.rng.choice(
                (0x00, NES_INPUT_RIGHT, NES_INPUT_RIGHT | NES_INPUT_A)))
            warmup_log.append(c)
            env._nes.controller = c
            env._nes.step()
        env._nes.controller = 0

    env._warmup = logged_warmup

    results = []
    try:
        for ep in range(args.episodes):
            warmup_log.clear()
            env.reset(seed=1000 + ep)
            base = list(boot_inputs) + list(warmup_log) + [0x00]
            axe_frame = len(boot_inputs) + len(warmup_log)
            greedy = args.greedy or (ep == 0 and not args.no_greedy_first)
            if not greedy:
                torch.manual_seed(1000 + ep)
            r = play_one(net, env, base,
                         os.path.join(OUT, f"_p_ep{ep}.mp4"),
                         os.path.join(OUT, f"_a_ep{ep}.mp4"),
                         args.scale, args.agent_scale, greedy, 1000 + ep, None)
            r["axe_frame"] = axe_frame
            results.append(r)
            print(f"  ep{ep + 1}: {'CLEAR!' if r['win'] else 'died':<6} "
                  f"{r['best']:7.0f} px   {r['frames']:5d} frames "
                  f"({r['frames'] / FPS:.0f}s)"
                  f"   {'greedy' if greedy else 'sampled'}", flush=True)
            if r["win"]:
                break
    finally:
        env.close()
        if os.path.exists(state_path):
            os.remove(state_path)

    best = max(results, key=lambda r: (r["win"], r["best"]))
    tag = args.tag
    p_final = os.path.join(OUT, f"v{tag}_player.mp4")
    a_final = os.path.join(OUT, f"v{tag}_agent.mp4")
    for src, dst in ((best["player"], p_final), (best["agent"], a_final)):
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)
    for r in results:
        for p in (r["player"], r["agent"]):
            if p not in (best["player"], best["agent"]) and os.path.exists(p):
                os.remove(p)

    meta = {
        "checkpoint": args.checkpoint,
        "stage": f"1-{args.start_round + 1}",
        "fps_video": FPS,
        "fps_nes": 60.098804,
        "sample_rate": 44100,
        "inputs_from_power_on": True,
        "pokes": [{"from": poke_range[0], "to": poke_range[1],
                   "addr": R.ROUND, "value": args.start_round}],
        "axe_frame": best["axe_frame"],
        "audio_start_frame": best["axe_frame"] + 1,
        "video_frames": best["frames"],
        "dump_every": DUMP_EVERY,
    }
    with open(os.path.join(OUT, f"v{tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(OUT, f"v{tag}_inputs.bin"), "wb") as fh:
        fh.write(bytes(best["inputs"]))
    with open(os.path.join(OUT, f"v{tag}_progress.json"), "w") as fh:
        json.dump(best["dumps"], fh, indent=1)

    print()
    print(f"cleared stage : {'YES' if best['win'] else 'no'}")
    print(f"farthest      : {best['best']:.0f} px")
    print(f"length        : {best['frames'] / FPS:.1f} s")
    print(f"player video  : {p_final} "
          f"({os.path.getsize(p_final) / 1e6:.1f} MB)")
    print(f"agent  video  : {a_final} "
          f"({os.path.getsize(a_final) / 1e6:.1f} MB)")
    print(f"inputs        : v{tag}_inputs.bin ({len(best['inputs'])} bytes)")
    print()
    print("next: node tools/jsnes_replay2.mjs <rom> "
          f"outputs/v{tag}_inputs.bin outputs/v{tag}_meta.json "
          f"outputs/v{tag}_audio")


if __name__ == "__main__":
    main()
