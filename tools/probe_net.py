"""Is the network actually reacting to what it sees?

Symptom: while a snail closes from +210 px to +30 px, the policy logits
do not move at all (right+jump = +3.28 the whole way).  That means the
agent is a constant function of its input -- it never throws because it
literally cannot tell that something is in front of it.

This walks the signal through the network, layer by layer:

  img diff     - do the stacked frames change at all?
  conv out     - does the CNN produce different features?
  h (fused)    - does the fused vector change?
  logits       - does the policy change?

then perturbs each input on its own (img -> zeros / noise, ram -> zeros)
to see which branch the output actually depends on.

    python tools/probe_net.py <checkpoint> [steps]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import OBS_DIM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic  # noqa: E402


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n = int(ck["model"]["policy.weight"].shape[0])
    net = ActorCritic(OBS_DIM, n, 128)
    net.load_state_dict(ck["model"], strict=False)
    net.eval()
    return net, n


def fwd(net, ram, img):
    with torch.no_grad():
        x = net.conv(img.float() / 255.0)
        h = net.fuse(torch.cat([x, ram], dim=-1))
        return x, h, net.policy(h)


def stats(name: str, a: np.ndarray, b: np.ndarray | None = None) -> None:
    if b is None:
        print(f"  {name:<10} mean {a.mean():8.4f}  std {a.std():8.4f}  "
              f"min {a.min():8.3f}  max {a.max():8.3f}")
    else:
        d = np.abs(a - b)
        print(f"  {name:<10} |Δ| mean {d.mean():8.5f}  max {d.max():8.5f}  "
              f"({(d > 1e-6).mean():.1%} of values differ)")


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_v6/ppo_00020480.pt"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120

    net, n_act = load(ckpt)
    env = AdventureIslandEnv(seed=0, start_with_axe=True)
    obs = env.reset(seed=0)

    snaps = []
    for i in range(n_steps):
        if i % 40 == 0:
            snaps.append((i, obs["ram"].copy(), obs["img"].copy()))
        with torch.no_grad():
            t = {"ram": torch.as_tensor(obs["ram"], dtype=torch.float32)
                 .unsqueeze(0),
                 "img": torch.as_tensor(obs["img"], dtype=torch.uint8)
                 .unsqueeze(0)}
            a = int(net(t)[0].argmax())
        obs, _, done, _ = env.step(a)
        if done:
            break
    env.close()

    print(f"{ckpt} ({n_act} actions), {len(snaps)} snapshots\n")
    if len(snaps) < 2:
        print("not enough snapshots - episode ended too early")
        return

    # --- does anything move between two distant moments? ---------------
    (i0, r0, im0), (i1, r1, im1) = snaps[0], snaps[-1]
    print(f"step {i0} vs step {i1}:")
    stats("ram", r0, r1)
    stats("img", im0.astype(np.float32), im1.astype(np.float32))

    x0, h0, l0 = fwd(net, torch.as_tensor(r0).unsqueeze(0),
                     torch.as_tensor(im0).unsqueeze(0))
    x1, h1, l1 = fwd(net, torch.as_tensor(r1).unsqueeze(0),
                     torch.as_tensor(im1).unsqueeze(0))
    stats("conv out", x0.numpy(), x1.numpy())
    stats("fused h", h0.numpy(), h1.numpy())
    stats("logits", l0.numpy(), l1.numpy())

    # --- how big are the two branches, and which one matters? ----------
    print("\nperturbation of the LAST snapshot:")
    base = l1.numpy().ravel()
    print(f"  baseline logits   {np.round(base, 3)}")

    for name, ram_t, img_t in [
        ("ram -> 0", torch.zeros_like(torch.as_tensor(r1).unsqueeze(0)),
         torch.as_tensor(im1).unsqueeze(0)),
        ("img -> 0", torch.as_tensor(r1).unsqueeze(0),
         torch.zeros_like(torch.as_tensor(im1).unsqueeze(0))),
        ("img -> noise", torch.as_tensor(r1).unsqueeze(0),
         torch.randint(0, 256, im1.shape).unsqueeze(0)),
    ]:
        _, _, l = fwd(net, ram_t, img_t)
        d = np.abs(l.numpy().ravel() - base)
        print(f"  {name:<12} max |Δlogit| {d.max():8.4f}   "
              f"argmax {ACT(l.numpy().ravel())}")

    # --- how much does the IMAGE branch alone vary over time? ----------
    xs = []
    for _, r, im in snaps:
        x, _, _ = fwd(net, torch.as_tensor(r).unsqueeze(0),
                      torch.as_tensor(im).unsqueeze(0))
        xs.append(x.numpy().ravel())
    xs = np.array(xs)
    print(f"\nconv features across {len(xs)} snapshots: "
          f"per-feature std over time = {xs.std(0).mean():.6f} "
          f"(0 = the CNN is blind)")


def ACT(v: np.ndarray) -> int:
    from env import ACTIONS
    return ACTIONS[int(v.argmax())][1]


if __name__ == "__main__":
    main()
