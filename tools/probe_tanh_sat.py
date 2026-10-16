"""Replay the pre-LayerNorm fusion layer and measure its Tanh saturation.

The bug (found 2026-10-13, see the comment above `self.fuse` in train.py): the
fusion layer was Linear + Tanh and its pre-activations ran to |x| ~ 6, so
98-100 % of the units sat at +-1 and the policy became a constant function of
its input -- a snail could close from 210 px to 30 px without the logits
moving by even 0.001.

The checkpoint the diagnosis used is runs/ppo_v6_pre_layernorm/ppo_00020480.pt
(the v6 continuation, 20,480 steps after the 378,880-step run).  This rebuilds
the OLD fuse (Linear + Tanh) from its weights and records, per decision step:

  * all 128 pre-activation values  -> histogram for the s05 visual
  * the policy logits              -> how little they move over the episode
  * input perturbations at the end -> ram -> 0, img -> 0, img -> noise

and writes them to video/src/tanh_data.json for the Remotion component.

    python tools/probe_tanh_sat.py [checkpoint] [steps]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env import (ACTIONS, IMG_H, IMG_STACK, IMG_W, OBS_DIM,  # noqa: E402
                 AdventureIslandEnv)
from train import ActorCritic  # noqa: E402

HIST_MIN, HIST_MAX, HIST_BINS = -8.0, 8.0, 64
SAT_X = 3.0        # |x| > 3   ->  |tanh| > 0.995
HARD_X = 4.5       # |x| > 4.5 ->  |tanh| > 0.99975


def load_old_net(path: str):
    """ActorCritic carrying the ORIGINAL Linear+Tanh fuse, plus the raw Linear."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    if "fuse.1.weight" in sd:
        raise SystemExit(f"{path} is post-LayerNorm -- nothing to measure")
    n_act = int(sd["policy.weight"].shape[0])
    net = ActorCritic(OBS_DIM, n_act, 128)
    net.load_state_dict(sd, strict=False)
    net.eval()
    with torch.no_grad():
        n_flat = net.conv(torch.zeros(1, IMG_STACK, IMG_H, IMG_W)).shape[1]
    in_ck = int(sd["fuse.0.weight"].shape[1])
    obs_ck = in_ck - n_flat
    if obs_ck <= 0:
        raise SystemExit(f"{path}: conv flatten {n_flat} does not fit fuse "
                         f"in_features {in_ck} -- image dims changed?")
    if obs_ck != OBS_DIM:
        print(f"note: checkpoint wants {obs_ck}-dim RAM, env gives {OBS_DIM}")
    lin = nn.Linear(in_ck, 128)
    with torch.no_grad():
        lin.weight.copy_(sd["fuse.0.weight"])
        lin.bias.copy_(sd["fuse.0.bias"])
    train_steps = None
    for k in ("steps", "global_step", "step", "timesteps", "total_steps"):
        if k in ck:
            train_steps = int(ck[k])
            break
    if train_steps is None:
        digits = "".join(ch for ch in os.path.basename(path) if ch.isdigit())
        train_steps = int(digits) if digits else None
    return net, lin, n_act, obs_ck, train_steps


def fit_ram(ram: np.ndarray, obs_ck: int) -> np.ndarray:
    ram = np.asarray(ram, dtype=np.float32).ravel()
    if len(ram) == obs_ck:
        return ram
    out = np.zeros(obs_ck, dtype=np.float32)
    n = min(len(ram), obs_ck)
    out[:n] = ram[:n]
    return out


def main() -> None:
    ckpt = (sys.argv[1] if len(sys.argv) > 1
            else "runs/ppo_v6_pre_layernorm/ppo_00020480.pt")
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    ck_path = ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)

    net, lin, n_act, obs_ck, train_steps = load_old_net(ck_path)
    acts = [name for _, name in ACTIONS][:n_act]
    print(f"{ckpt}  ({train_steps} steps, {n_act} actions, RAM {obs_ck}-dim)")

    env = AdventureIslandEnv(seed=0, start_with_axe=True)
    obs = env.reset(seed=0)

    pres, logits, mean_trace = [], [], []
    r_last = fit_ram(obs["ram"], obs_ck)
    im_last = np.asarray(obs["img"], dtype=np.uint8).copy()
    for _ in range(steps):
        r_last = fit_ram(obs["ram"], obs_ck)
        im_last = np.asarray(obs["img"], dtype=np.uint8).copy()
        with torch.no_grad():
            img = torch.as_tensor(im_last, dtype=torch.float32).unsqueeze(0) / 255.0
            ram = torch.as_tensor(r_last, dtype=torch.float32).unsqueeze(0)
            x = net.conv(img)
            pre = lin(torch.cat([x, ram], dim=-1))
            lg = net.policy(torch.tanh(pre))
        pres.append(pre.numpy().ravel())
        logits.append(lg.numpy().ravel())
        mean_trace.append(float(np.abs(pre.numpy()).mean()))
        a = int(lg.numpy().ravel().argmax())
        obs, _, done, _ = env.step(a)
        if done:
            break
    env.close()

    pres = np.asarray(pres)          # (steps, 128) pre-activations
    logits = np.asarray(logits)      # (steps, n_act) policy logits
    flat = pres.ravel()
    a_flat = np.abs(flat)

    counts, _ = np.histogram(flat, bins=HIST_BINS, range=(HIST_MIN, HIST_MAX))
    under = int((flat < HIST_MIN).sum())
    over = int((flat > HIST_MAX).sum())

    def lg_of(ram_v: np.ndarray, img_v: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            img = torch.as_tensor(img_v, dtype=torch.float32).unsqueeze(0) / 255.0
            ram = torch.as_tensor(ram_v, dtype=torch.float32).unsqueeze(0)
            pre = lin(torch.cat([net.conv(img), ram], dim=-1))
            return net.policy(torch.tanh(pre)).numpy().ravel()

    base = logits[-1]
    rng = np.random.default_rng(0)
    pert = {
        "ram0": float(np.abs(lg_of(np.zeros_like(r_last), im_last) - base).max()),
        "img0": float(np.abs(lg_of(r_last, np.zeros_like(im_last)) - base).max()),
        "noise": float(np.abs(lg_of(r_last, rng.integers(0, 256, im_last.shape,
                                                       dtype=np.uint8)) - base).max()),
    }

    data = {
        "checkpoint": ckpt.replace("\\", "/"),
        "train_steps": train_steps,
        "n_units": int(pres.shape[1]),
        "n_actions": int(logits.shape[1]),
        "actions": acts,
        "steps_measured": int(pres.shape[0]),
        "samples": int(flat.size),
        "mean_abs": float(a_flat.mean()),
        "median_abs": float(np.median(a_flat)),
        "p95_abs": float(np.percentile(a_flat, 95)),
        "max_abs": float(a_flat.max()),
        "pct_sat": float((a_flat > SAT_X).mean() * 100.0),
        "pct_sat_hard": float((a_flat > HARD_X).mean() * 100.0),
        "sat_x": SAT_X,
        "hard_x": HARD_X,
        "hist_min": HIST_MIN,
        "hist_max": HIST_MAX,
        "hist_bins": [int(c) for c in counts],
        "hist_under": under,
        "hist_over": over,
        "logit_step_max_delta": float(np.abs(np.diff(logits, axis=0)).max())
        if len(logits) > 1 else 0.0,
        "logit_first_last_delta": float(np.abs(logits[-1] - logits[0]).max()),
        "logit_trace": [[round(float(v), 4) for v in row] for row in logits],
        "mean_abs_trace": [round(v, 3) for v in mean_trace],
        "perturb": pert,
    }

    out = os.path.join(ROOT, "video", "src", "tanh_data.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")

    print(f"steps measured      {data['steps_measured']}  "
          f"({data['samples']} pre-activation samples)")
    print(f"|x| mean {data['mean_abs']:.2f}  median {data['median_abs']:.2f}  "
          f"p95 {data['p95_abs']:.2f}  max {data['max_abs']:.2f}")
    print(f"saturated (|x|>{SAT_X})  {data['pct_sat']:.1f} %   "
          f"hard (|x|>{HARD_X})  {data['pct_sat_hard']:.1f} %")
    print(f"logits: max |delta| per step {data['logit_step_max_delta']:.5f}   "
          f"first vs last {data['logit_first_last_delta']:.5f}")
    print("logits per action (min .. max over the episode):")
    for j, name in enumerate(acts):
        col = logits[:, j]
        print(f"  {j} {name:<18} {col.min():+.3f} .. {col.max():+.3f}"
              f"   span {col.max() - col.min():.5f}")
    print(f"perturbation: ram->0 {pert['ram0']:.5f}   img->0 {pert['img0']:.5f}   "
          f"img->noise {pert['noise']:.5f}")
    print(f"histogram under/over {under}/{over}   -> {out}")


if __name__ == "__main__":
    main()
