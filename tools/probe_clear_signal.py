"""Is $007A really "stage complete"?

Two observations that do not add up:

  * tools/probe_14_final.py walked 1-3 under the invincibility cheat and
    $007A fired at f=5183 with progress **-33** -- i.e. the runner had not
    gone anywhere near the end of the stage.
  * env.py treats DEATH != 0 && $007A == 0 as "dying", which implies
    $007A can also be non-zero while dying.

If $007A fires on things other than a real stage clear, CLEAR_GAIN (and
`--clear-ends-episode`) is teaching the policy the wrong thing.  So watch
$007A frame by frame on a stage that is known to clear (1-3, model
models/ppo_1-3_clear_16M.pt) and print every transition with full context.

    python tools/probe_clear_signal.py --stage 1-3
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
from env import AdventureIslandEnv                         # noqa: E402

RAMP = " .:-=+*#%@"


def ascii_art(frame: np.ndarray, w: int = 76, h: int = 30) -> str:
    g = frame.astype(np.float32).mean(axis=2)
    H, W = g.shape
    ys = np.linspace(0, H, h + 1).astype(int)
    xs = np.linspace(0, W, w + 1).astype(int)
    return "\n".join(
        "".join(RAMP[min(len(RAMP) - 1,
                         int(g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
                             / 255.0 * len(RAMP)))]
                for j in range(w))
        for i in range(h))


def ids(nes):
    return [(s, int(nes[R.OBJ.id + R.OBJ.STRIDE * s]))
            for s in range(16)
            if int(nes[R.OBJ.id + R.OBJ.STRIDE * s])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="models/ppo_1-3_clear_16M.pt")
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=12000)
    args = ap.parse_args()

    ck = args.checkpoint
    if not os.path.exists(ck):
        print(f"!! no checkpoint {ck} -- running with random actions")
        ck = None

    env = AdventureIslandEnv(start_with_axe=True,
                             start_round=args.round,
                             stage_index=args.stage)
    if ck:
        env.attach_policy(ck) if hasattr(env, "attach_policy") else None

    print(f"checkpoint={ck} round={args.round} stage={args.stage}")
    if ck is None:
        print("(no policy attach API -- falling back to scripted run)")

    for ep in range(args.episodes):
        env.reset(seed=ep)
        nes = env._nes
        cam = env._cam
        prev = int(nes[R.STAGE_COMPLETE])
        hist = []
        print(f"\n--- episode {ep} ---")
        for f in range(args.max_frames):
            # scripted: hold right + hop; good enough to move forward
            nes.controller = 0x01 | (0x80 if f % 22 < 4 else 0)
            frame = np.asarray(nes.step(1), dtype=np.uint8)
            cur = int(nes[R.STAGE_COMPLETE])
            hist.append((cur, int(nes[R.DEATH]), int(nes[R.LIVES]),
                         cam.progress(nes)))
            if cur != prev:
                print(f"  f={f:6d}  $007A {prev} -> {cur}   "
                      f"DEATH={int(nes[R.DEATH])} LIVES={int(nes[R.LIVES])} "
                      f"prog={cam.progress(nes):.0f} "
                      f"STA={int(nes[R.STAMINA])}", flush=True)
                print("   last 12 frames (007A,DEATH,LIVES,prog):")
                for h in hist[-12:]:
                    print(f"     {h[0]:3d} {h[1]:3d} {h[2]:3d} {h[3]:7.0f}")
                print(ascii_art(frame))
                prev = cur
            if cur != 0:
                # keep going 120 more frames to see what follows
                for k in range(120):
                    nes.controller = 0
                    frame = np.asarray(nes.step(1), dtype=np.uint8)
                print(f"   +120 frames later: $007A="
                      f"{int(nes[R.STAGE_COMPLETE])} "
                      f"DEATH={int(nes[R.DEATH])} "
                      f"LIVES={int(nes[R.LIVES])} "
                      f"prog={cam.progress(nes):.0f} ids={ids(nes)}")
                print(ascii_art(frame))
                break
        else:
            print(f"  no $007A transition in {args.max_frames} frames; "
                  f"final prog={cam.progress(nes):.0f} "
                  f"LIVES={int(nes[R.LIVES])}")


if __name__ == "__main__":
    main()
