"""Grab a save state at the very start of a given stage.

Naming: the round register $0038 is 0-based -- stage 1-1 plays with
``round = 0`` and the game bumps it to 1 the moment stage 1-1 is cleared
(tools/probe_round2_natural.py: "--> ROUND 1 at +687f").  So:

    --round 0   stage 1-1  (第一关)
    --round 1   stage 1-2  (第二关)
    --round 2   stage 1-3  (第三关)
    ...

Two ways to get there, both produced here:

  warp     set $0038 = N while the game initialises stage 1-1 (right after
           the title-screen START press).  Verified: the game reads the
           register at stage init and loads a different layout -- spawn
           y is 152 for round 0, 80 for round 1, 192 for round 2.
  natural  let the trained greedy policy actually CLEAR the previous stages
           and step through the tally -> banner -> gameplay transition.
           Only stage 1-1 has a model, so this only works for --round 1.

"Gameplay has begun" is detected behaviourally, not by counting frames:
hold RIGHT for 30 frames on a copy of the state and watch the player x.
The first frame where the player actually moves is the save point.

Both states are dumped and compared pixel-wise, which tells us whether the
warp lands on the same stage the game would naturally reach.

usage:
    python tools/make_round2_state.py                  # both + compare
    python tools/make_round2_state.py --warp-only      # no model needed
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from cynes import NES, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, AdventureIslandEnv, ROM  # noqa: E402
from train import ActorCritic  # noqa: E402

SAVE_DIR = os.path.join(_ROOT, "saves")
SHOT_DIR = os.path.join(_ROOT, "outputs", "probe_round2")
WANTED_ROUND = 1                      # 0-based: 1 == the second stage
PROBE_FRAMES = 30                     # frames of "hold right" to test input
PROBE_MIN_PX = 5.0                    # ... and how far that must move us


def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SHOT_DIR, exist_ok=True)


def player_xy(nes):
    return (R.px(R.read_obj_world_x(nes, R.PLAYER_SLOT)),
            R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT)))


def control_live(nes):
    """True if the player is under OUR control right now (state restored).

    Two conditions, because one is not enough:

      * idle for 30 frames: the player must NOT move on its own.  Right
        after a stage clear the game walks Higgins to the right by itself,
        so "holding RIGHT moved him" is a false positive there.
      * holding RIGHT for 30 frames: the player must actually move.

    Both probes run on a copy: the emulator is rolled back afterwards.
    """
    st = nes.save()
    x0, _ = player_xy(nes)
    nes.controller = 0
    for _ in range(PROBE_FRAMES):
        nes.step()
    x_idle, _ = player_xy(nes)
    nes.load(st)

    nes.controller = 0x01
    for _ in range(PROBE_FRAMES):
        nes.step()
    nes.controller = 0
    x_right, _ = player_xy(nes)
    nes.load(st)
    return abs(x_idle - x0) < 2.0 and (x_right - x0) > PROBE_MIN_PX


def wait_for_picture(nes, max_frames=400):
    """Step until the stage has faded in (the frame is not black any more).

    Right after a stage load the PPU is off for ~180 frames: the game
    logic runs and the player already answers the controller, but the
    screen is pure black.  A state saved there starts every episode with
    ~45 black observations and looks like "still loading" in a screenshot.
    """
    for i in range(max_frames):
        nes.controller = 0
        if float(np.asarray(nes.step()).mean()) > 30.0:
            print(f"  picture on at +{i}f "
                  f"(mean brightness {np.asarray(nes.step()).mean():.0f})")
            return i
    return None


def wait_for_control(nes, max_frames, every=10, tag=""):
    """Step until the player reacts to input.  Returns the frame index."""
    for i in range(max_frames):
        if control_live(nes):
            print(f"  {tag}control is live at +{i}f "
                  f"(area={nes[R.AREA]} round={nes[R.ROUND]} "
                  f"ckpt={nes[R.CHECKPOINT]} lives={nes[R.LIVES]} "
                  f"sta={nes[R.STAMINA]} px={player_xy(nes)})")
            return i
        for _ in range(every):
            nes.controller = 0
            nes.step()
    return None


def snapshot(nes, tag):
    """Frame + state of the current moment (also reports the brightness)."""
    state = nes.save()
    # .copy() is mandatory: np.asarray(nes.step()) is a VIEW into the
    # emulator's frame buffer, so it goes stale (and dangling) the moment
    # the emulator steps again or is deleted.
    arr = np.asarray(nes.step(), dtype=np.uint8).copy()
    print(f"  snapshot {tag}: mean brightness {arr.mean():.1f}")
    Image.fromarray(arr, mode="RGB").resize((512, 480), Image.NEAREST).save(
        os.path.join(SHOT_DIR, f"{tag}.png"))
    return state, arr


def dump(state, name):
    path = os.path.join(SAVE_DIR, name)
    with open(path, "wb") as fh:
        fh.write(state)
    print(f"  saved {path} ({len(state)} bytes)")
    return path


# ----------------------------------------------------------------------
def build_warp(round_value=WANTED_ROUND):
    """Title -> START with $0038 forced, then wait for control."""
    tag = f"round{round_value + 1}"
    print(f"=== warp: force ROUND={round_value} during stage init ===")
    nes = NES(ROM)
    nes.step(frames=300)                       # title screen
    nes.controller = NES_INPUT_START
    for _ in range(90):
        nes[R.ROUND] = round_value             # hammer it through init
        nes.step()
    nes.controller = 0
    i = wait_for_control(nes, 600, every=10, tag="warp  ")
    if i is None:
        print("  !! never got control - warp failed")
        del nes
        return None
    wait_for_picture(nes)
    if not control_live(nes):
        print("  !! lost control while the stage faded in")
        del nes
        return None
    state, arr = snapshot(nes, f"warp_{tag}_start")
    dump(state, f"{tag}_start_warp.state")
    del nes
    return state, arr


def build_natural(ckpt, round_value=WANTED_ROUND):
    """Play the previous stages for real, then walk the transition.

    Only stage 1-1 has a trained model, so this is really only usable for
    --round 1; anything else is reported as unsupported.
    """
    if round_value != 1:
        print("\n=== natural: skipped (no model for the earlier stages "
              "of round %d) ===" % round_value)
        return None
    print("\n=== natural: clear stage 1-1 with the trained policy ===")
    net = ActorCritic(OBS_DIM, len(ACTIONS))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    own = net.state_dict()
    keep = {k: v for k, v in ck["model"].items()
            if k in own and own[k].shape == v.shape}
    net.load_state_dict(keep, strict=False)
    net.eval()

    env = AdventureIslandEnv()
    nes = env._nes
    try:
        for attempt in range(6):
            print(f"  attempt {attempt + 1}/6 ...", flush=True)
            out = _try_clear(env, nes, net)
            if out is not None:
                state, arr = out
                dump(state, "round2_start_natural.state")
                return state, arr
        print("  !! never cleared stage 1-1 in 6 attempts")
        return None
    finally:
        env.close()


def _try_clear(env, nes, net):
    """Play one episode; on a clear, walk the transition and grab the state."""
    env.reset()
    cleared_at = None
    for f in range(60 * 300):
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
            print(f"  stage 1-1 cleared at frame {f} ({f / 60:.1f}s)")
            break
        if nes[R.DEATH] != 0:
            print(f"  died at frame {f} - retrying")
            return None
    if cleared_at is None:
        print("  ran out of time - retrying")
        return None

    # walk the tally -> banner -> gameplay transition
    i = wait_for_control(nes, 60 * 40, every=10, tag="nature")
    if i is None:
        print("  !! never got control after the clear")
        return None
    wait_for_picture(nes)
    return snapshot(nes, "natural_round2_start")


def compare(a, b, label_a="warp", label_b="natural"):
    """Pixel diff of the two first frames of gameplay."""
    if a is None or b is None:
        print("\n(comparison skipped - one state is missing)")
        return
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    frac = float((diff > 24).mean())
    print(f"\n=== compare {label_a} vs {label_b} ===")
    print(f"  pixels differing >+/-24 of 256: {frac * 100:.2f}%")
    print(f"  mean |diff|: {diff.mean():.2f}")
    # the HUD (top 24 rows) legitimately differs (score); compare the
    # play field below it
    field = diff[24:, :]
    print(f"  play field only (row 24+): {(field > 24).mean() * 100:.2f}%")
    if (field > 24).mean() < 0.05:
        print("  verdict: SAME STAGE - the warp lands on the real round 2")
    else:
        print("  verdict: DIFFERENT - do not use the warp state")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=WANTED_ROUND,
                    help="0-based round register value: 0 = stage 1-1, "
                         "1 = stage 1-2, 2 = stage 1-3, ...")
    ap.add_argument("--warp-only", action="store_true",
                    help="skip the (slow) natural playthrough")
    ap.add_argument("--ckpt", default=os.path.join(
        _ROOT, "runs", "ppo_v10", "ppo_final.pt"))
    args = ap.parse_args()

    ensure_dirs()
    warped = build_warp(args.round)
    natural = None if args.warp_only else build_natural(args.ckpt, args.round)
    compare(warped[1] if warped else None,
            natural[1] if natural else None)


if __name__ == "__main__":
    main()
