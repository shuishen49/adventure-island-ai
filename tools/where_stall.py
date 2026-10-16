"""Where does the policy get stuck?

The episode can end three ways (see env.py):
  * death   - ran into something lethal (lives dropped)
  * stall   - no new best progress for STALL_STEPS (250) decisions  <-- the
              interesting one: the agent walks up to some spot and then
              refuses (or fails) to go further
  * timeout - MAX_STEPS (4500) reached while still making progress

This drives the *current* checkpoint through N episodes, records which way
each one ended and at which pixel, prints a histogram of the stall spots and
writes an annotated screenshot of the frame where it gave up.

usage:  python tools/where_stall.py [ckpt] [episodes]
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ram_map as R  # noqa: E402
from env import (ACTIONS, OBS_DIM, AdventureIslandEnv, MAX_STEPS,  # noqa: E402
                 STALL_STEPS)
from train import ActorCritic, newest_checkpoint  # noqa: E402
from visualize import annotate, objects  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs")
CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    newest_checkpoint("runs/ppo_v9") or "runs/ppo/ppo_final.pt")
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def load(path):
    net = ActorCritic(OBS_DIM, len(ACTIONS))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = net.state_dict()
    usable = {k: v for k, v in ck["model"].items()
              if k in sd and sd[k].shape == v.shape}
    dropped = [k for k in ck["model"] if k not in usable]
    net.load_state_dict(usable, strict=False)
    net.eval()
    if dropped:
        print(f"  (dropped {len(dropped)} tensor(s): {dropped})")
    return net


def act(net, obs):
    # NOTE: ActorCritic.forward() already does `img.float() / 255.0`.
    # Handing it a pre-divided tensor squashes every pixel to ~0.004 and
    # blinds the conv branch -- the agent then behaves far worse than it
    # does during training.  Pass the raw uint8 stack.
    img = torch.as_tensor(obs["img"][None])
    ram = torch.as_tensor(obs["ram"][None], dtype=torch.float32)
    with torch.no_grad():
        logits, _ = net({"img": img, "ram": ram})
        a = int(torch.distributions.Categorical(logits=logits).sample())
    return a


def main():
    net = load(CKPT)
    env = AdventureIslandEnv()
    print(f"driven by {CKPT}")
    print(f"MAX_STEPS={MAX_STEPS}  STALL_STEPS={STALL_STEPS}\n")

    stalls, deaths, timeouts = [], [], []
    act_hist = {}

    for ep in range(N_EP):
        obs = env.reset()
        hist = []          # (step, best_progress, action)
        best = -1e9
        last_best_step = 0
        cause = "timeout"
        snap_ep = None
        for t in range(MAX_STEPS):
            a = act(net, obs)
            act_hist[a] = act_hist.get(a, 0) + 1
            obs, r, done, info = env.step(a)
            bp = info["best_progress"]
            if bp > best:
                best, last_best_step = bp, t
            hist.append((t, bp, a))
            if done:
                if info.get("death"):
                    cause = "death"
                elif t - last_best_step >= STALL_STEPS:
                    cause = "stall"
                else:
                    cause = "timeout"
                break
        # where did it stop making progress?
        if cause == "stall":
            stalls.append(best)
        elif cause == "death":
            deaths.append(best)
            snap_ep = len(hist) - 1      # frame of the fatal contact
        else:
            timeouts.append(best)
            snap_ep = len(hist) - 1
        print(f"  ep{ep + 1}: {cause:<8} best {best:7.0f} px "
              f"x_prog {info.get('x_progress', 0):7.0f} px "
              f"({len(hist)} steps)")
        if ep == 0:
            print(f"     (progress right after reset: {hist[0][1]:.0f} px)")

        # annotated screenshot of the frame where the episode ended
        if snap_ep is not None and getattr(env, "_last_frame", None) is not None:
            tag = "stall" if cause == "stall" else cause
            ann = annotate(np.asarray(env._last_frame), env._nes,
                           ACTIONS[hist[snap_ep][2]])
            Image.fromarray(ann).save(os.path.join(OUT, f"where_{tag}.png"))
            objs = objects(env._nes)
            print(f"     -> wrote where_{tag}.png "
                  f"({len(objs)} objects on that frame)")

    print()
    print(f"stalls  : {[round(x) for x in stalls]}")
    print(f"deaths  : {[round(x) for x in deaths]}")
    print(f"timeouts: {[round(x) for x in timeouts]}")
    if stalls:
        a = np.array(stalls)
        print(f"\nstall spots: median {np.median(a):.0f} px  "
              f"min {a.min():.0f}  max {a.max():.0f}")
        hist, edges = np.histogram(a, bins=[0, 1000, 2000, 2500, 3000, 3500,
                                            4000, 5000, 6000, 99999])
        for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
            if h:
                print(f"   {lo:>6}-{hi if hi < 99999 else '+inf':>6} px : "
                      f"{'#' * int(h)} {h}")
    tot = sum(act_hist.values()) or 1
    print("\naction mix over the whole run:")
    for a, c in sorted(act_hist.items(), key=lambda kv: -kv[1]):
        name = ACTIONS[a][1] if isinstance(ACTIONS[a], (tuple, list)) \
            else str(ACTIONS[a])
        print(f"   {a} {name:<18} {100 * c / tot:5.1f}%")
    print(f"\nwrote {os.path.join(OUT, 'where_stall.png')}")
    env.close()


if __name__ == "__main__":
    main()
