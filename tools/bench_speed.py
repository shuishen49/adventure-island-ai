"""Where does the training time actually go?

Training reports ~32 env-steps/s.  Each step is
    (a) 8 environments advancing 4 emulator frames + building the obs
    (b) one forward pass of a small CNN with batch 8
This measures both, and checks whether torch's thread count helps or hurts.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from cynes import NES  # noqa: E402

from env import ACTIONS, OBS_DIM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic  # noqa: E402

N_ENV = 8
N_STEP = 120


def bench_env():
    """Cost of the emulator + observation, no network at all."""
    envs = [AdventureIslandEnv(seed=s, start_with_axe=True) for s in range(N_ENV)]
    for e in envs:
        e.reset(seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(N_STEP):
        for e in envs:
            e.step(int(rng.integers(0, len(ACTIONS))))
    dt = time.perf_counter() - t0
    for e in envs:
        e.close()
    steps = N_STEP * N_ENV
    return dt, steps / dt, steps * 4 / dt      # env-steps/s, game-frames/s


def bench_net(threads: int, img=None, ram=None, n_rep: int = 60):
    torch.set_num_threads(threads)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_rep):
            net({"ram": ram, "img": img})
    dt = (time.perf_counter() - t0) / n_rep
    return dt, 1.0 / dt


def main():
    print(f"cpu_count = {os.cpu_count()}   "
          f"torch threads now = {torch.get_num_threads()}")

    dt_env, eps, gfps = bench_env()
    print(f"\n[emulator + obs]  {N_ENV} envs x {N_STEP} steps")
    print(f"   {dt_env:6.2f} s  ->  {eps:6.1f} env-steps/s   "
          f"{gfps:6.0f} game-frames/s   ({1000*dt_env/(N_STEP*N_ENV):.2f} ms/step)")

    # a realistic batch, taken from a live environment
    e = AdventureIslandEnv(seed=0, start_with_axe=True)
    obs = e.reset(seed=0)
    ram = torch.as_tensor(
        np.stack([obs["ram"]] * N_ENV), dtype=torch.float32)
    img = torch.as_tensor(
        np.stack([obs["img"]] * N_ENV), dtype=torch.uint8)
    e.close()

    print("\n[network forward] batch = 8, no grad")
    for th in (1, 2, 4, 8, 16, 56):
        if th > os.cpu_count():
            continue
        dt, fps = bench_net(th, img, ram)
        print(f"   threads {th:3d}: {1000*dt:7.2f} ms   {fps:7.0f} fwd/s   "
              f"-> ceiling {fps * N_ENV / 1:8.0f} env-steps/s (if net-only)")

    # --- a full learning step (forward + backward), which is what PPO pays
    print("\n[PPO update] forward+backward, batch 128 x 4 epochs")
    for th in (1, 2, 4, 8, 56):
        if th > os.cpu_count():
            continue
        torch.set_num_threads(th)
        net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)
        ram_b = torch.randn(128, OBS_DIM)
        img_b = torch.zeros(128, 2, 80, 86, dtype=torch.uint8)
        t0 = time.perf_counter()
        for _ in range(4):
            logits, v = net({"ram": ram_b, "img": img_b})
            loss = (logits.pow(2).mean() + (v - 1).pow(2).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        dt = time.perf_counter() - t0
        print(f"   threads {th:3d}: {dt:7.2f} s per update")


if __name__ == "__main__":
    main()
