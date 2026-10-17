"""PPO training for Adventure Island on CPU.

Everything is deliberately small: a 27-dim RAM feature vector (no pixels)
and a 2-layer MLP, so training runs comfortably on a CPU core or two.

    python train.py --timesteps 500000 --nenvs 8
    python train.py --help
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env import (AdventureIslandEnv, N_ACTIONS, OBS_DIM,
                 IMG_H, IMG_W, IMG_STACK)

# Measured with tools/bench_speed.py on this 112-core box: for a network this
# small, intra-op threading is pure overhead -- the forward pass was 2x SLOWER
# with 56 threads than with 1 (2.62 ms vs 1.30 ms) and a PPO update 5x slower
# (0.35 s vs 0.07 s).  Parallelism now comes from worker processes.
torch.set_num_threads(min(8, os.cpu_count()))


# ----------------------------------------------------------------------
# network
# ----------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Small CNN over the frame stack, fused with the RAM feature vector.

    The convolution branch is what lets the agent see platforms and pits;
    the RAM branch keeps the exact numbers (velocity, gravity, stamina).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 img_hw: tuple[int, int] = (IMG_H, IMG_W),
                 img_stack: int = IMG_STACK) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_stack, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, img_stack, *img_hw)).shape[1]
        # 2026-10-13 (bugfix): this used to be Linear + Tanh, and it was
        # SATURATED -- measured pre-activation |x| ~5.8, so 98-100 % of the
        # units sat at +-1 and the whole net degenerated into a sign
        # function: with a snail closing from 210 px to 30 px the logits
        # moved by 0.001 and zeroing the RAM vector moved them by 0.0001.
        # The agent was effectively blind to its own inputs.  LayerNorm
        # (before the activation) keeps the pre-activation spread in a sane
        # range and ReLU has no upper bound, so the signal survives.
        self.fuse = nn.Sequential(
            nn.Linear(n_flat + obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, obs):
        img = obs["img"].float() / 255.0
        x = self.conv(img)
        h = self.fuse(torch.cat([x, obs["ram"]], dim=-1))
        return self.policy(h), self.value(h).squeeze(-1)

    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), value

    def evaluate(self, x, a):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        return dist.log_prob(a), dist.entropy(), value


# ----------------------------------------------------------------------
# rollout buffer
# ----------------------------------------------------------------------
@dataclass
class Rollout:
    ram: np.ndarray
    img: np.ndarray
    actions: np.ndarray
    logp: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray

    @classmethod
    def alloc(cls, steps: int, nenvs: int) -> "Rollout":
        return cls(
            ram=np.zeros((steps, nenvs, OBS_DIM), dtype=np.float32),
            img=np.zeros((steps, nenvs, IMG_STACK, IMG_H, IMG_W),
                         dtype=np.uint8),
            actions=np.zeros((steps, nenvs), dtype=np.int64),
            logp=np.zeros((steps, nenvs), dtype=np.float32),
            rewards=np.zeros((steps, nenvs), dtype=np.float32),
            dones=np.zeros((steps, nenvs), dtype=np.float32),
            values=np.zeros((steps, nenvs), dtype=np.float32),
        )


def collate(obs_list) -> dict[str, np.ndarray]:
    """list of {"ram", "img"} -> batched arrays."""
    return {"ram": np.stack([o["ram"] for o in obs_list]),
            "img": np.stack([o["img"] for o in obs_list])}


def as_tensor(batch) -> dict[str, torch.Tensor]:
    return {"ram": torch.as_tensor(batch["ram"], dtype=torch.float32),
            "img": torch.as_tensor(batch["img"], dtype=torch.uint8)}


def compute_gae(rew, dones, values, last_value, gamma=0.99, lam=0.95):
    T, N = rew.shape
    adv = np.zeros_like(rew)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rew[t] + gamma * next_v * nonterm - values[t]
        last_gae = delta + gamma * lam * nonterm * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


# ----------------------------------------------------------------------
# parallel rollout workers
# ----------------------------------------------------------------------
def _worker_loop(rank: int, conn, args, seed: int) -> None:
    """One emulator + one copy of the policy, in its own process.

    The collection loop used to advance all eight environments in a single
    process, so the emulator cost added up serially (~17 s per update) while
    112 cores sat idle.  Here each worker walks its own environment and runs
    its own forward passes; the parent only broadcasts weights and gathers
    the finished trajectories.
    """
    torch.set_num_threads(1)
    env = AdventureIslandEnv(seed=seed,
                             start_with_axe=not args.no_start_axe,
                             start_round=args.start_round,
                             start_state=args.start_state,
                             stage_pool=args.stage_pool,
                             stage_weights=args.stage_weights,
                             clear_ends_episode=args.clear_ends_episode)
    net = ActorCritic(OBS_DIM, N_ACTIONS, args.hidden)
    obs = env.reset(seed=seed)
    n_steps = args.steps_per_update
    try:
        while True:
            msg = conn.recv()
            if msg is None:                      # parent asked us to stop
                break
            net.load_state_dict(msg)
            net.eval()

            ram = np.zeros((n_steps, OBS_DIM), dtype=np.float32)
            img = np.zeros((n_steps, IMG_STACK, IMG_H, IMG_W),
                           dtype=np.uint8)
            actions = np.zeros(n_steps, dtype=np.int64)
            logp = np.zeros(n_steps, dtype=np.float32)
            rewards = np.zeros(n_steps, dtype=np.float32)
            dones = np.zeros(n_steps, dtype=np.float32)
            values = np.zeros(n_steps, dtype=np.float32)
            ep_x: list[float] = []
            n_ep = 0
            o = obs

            for t in range(n_steps):
                batch = {"ram": torch.from_numpy(o["ram"][None]),
                         "img": torch.from_numpy(o["img"][None])}
                with torch.no_grad():
                    a, lp, _, v = net.act(batch)
                act = int(a)
                ram[t] = o["ram"]
                img[t] = o["img"]
                actions[t] = act
                logp[t] = float(lp)
                values[t] = float(v)

                o, r, done, info = env.step(act)
                rewards[t] = r
                dones[t] = float(done)
                if done:
                    n_ep += 1
                    ep_x.append(float(info["x_progress"]))
                    o = env.reset()

            conn.send({"ram": ram, "img": img, "actions": actions,
                       "logp": logp, "rewards": rewards, "dones": dones,
                       "values": values,
                       "last_obs": {"ram": o["ram"], "img": o["img"]},
                       "n_ep": n_ep, "ep_x": ep_x})
    finally:
        env.close()
        conn.close()


class ParallelRollout:
    """Owns the worker processes and drives one collection round."""

    def __init__(self, args, seed: int) -> None:
        import multiprocessing as mp
        self.ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for i in range(args.workers):
            parent, child = self.ctx.Pipe(duplex=True)
            p = self.ctx.Process(target=_worker_loop,
                                 args=(i, child, args, seed + i),
                                 daemon=True)
            p.start()
            child.close()                        # only the parent writes
            self.conns.append(parent)
            self.procs.append(p)

    def collect(self, weights) -> list[dict]:
        for c in self.conns:
            c.send(weights)
        return [c.recv() for c in self.conns]

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except OSError:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def newest_checkpoint(folder: str) -> str | None:
    """Newest ``ppo_<steps>.pt`` in `folder`, by its recorded step count.

    Used by ``--resume`` to pick up exactly where a previous run stopped.
    """
    try:
        names = [f for f in os.listdir(folder)
                 if f.startswith("ppo_") and f.endswith(".pt")]
    except OSError:
        return None

    def step_of(name: str) -> int:
        try:
            return int(name[len("ppo_"):-len(".pt")])
        except ValueError:
            return -1

    names = [n for n in names if step_of(n) >= 0]
    if not names:
        return None
    return os.path.join(folder, max(names, key=step_of))


# ----------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------
def train(args) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    envs = []
    workers = None
    if args.workers > 0:
        # each worker is a process with its own emulator and policy copy
        workers = ParallelRollout(args, args.seed)
        print(f"parallel: {args.workers} worker processes, "
              f"{torch.get_num_threads()} intra-op threads", flush=True)
        obs_np = None
    else:
        envs = [AdventureIslandEnv(seed=args.seed + i,
                                   start_with_axe=not args.no_start_axe,
                                   start_round=args.start_round,
                                   start_state=args.start_state,
                                   stage_pool=args.stage_pool,
                                   stage_weights=args.stage_weights,
                                   clear_ends_episode=args.clear_ends_episode)
                for i in range(args.nenvs)]
        obs_np = collate([e.reset(seed=args.seed + i)
                          for i, e in enumerate(envs)])
    print(f"start-with-axe: {not args.no_start_axe}", flush=True)

    net = ActorCritic(OBS_DIM, N_ACTIONS, args.hidden).to(device)
    start_steps = 0
    if args.resume:
        ckpt = newest_checkpoint(args.save_dir)
        if ckpt:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            net.load_state_dict(ck["model"])
            start_steps = int(ck.get("steps", 0))
            print(f"resumed {ckpt} at {start_steps} env steps "
                  f"(continuing the run)", flush=True)
        else:
            print(f"--resume: no checkpoint in {args.save_dir}, "
                  f"starting from scratch", flush=True)
    elif args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        # Donate everything that still fits: a checkpoint from an older
        # action set (8 actions) gives us its convolution stack and value
        # head; only the policy head has to be re-learnt.  Keys whose
        # SHAPE changed are dropped first -- strict=False alone still
        # raises on a size mismatch.
        own = net.state_dict()
        keep = {k: v for k, v in ck["model"].items()
                if k in own and own[k].shape == v.shape}
        dropped = [k for k in ck["model"] if k not in keep]
        net.load_state_dict(keep, strict=False)
        start_steps = int(ck.get("steps", 0))
        print(f"warm-start from {args.init_from} "
              f"({start_steps} env steps)")
        print(f"  inherited {len(keep)} tensors, "
              f"re-initialised {dropped}")
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    buf = Rollout.alloc(args.steps_per_update, args.nenvs)
    steps_per_update = args.steps_per_update * args.nenvs
    log_path = os.path.join(args.save_dir, "progress.csv")
    append_log = args.resume and start_steps > 0 and os.path.exists(log_path)
    if append_log:
        # A previous run may have logged a few updates after its last saved
        # checkpoint.  Those rows cannot be reproduced, so drop them and keep
        # the resumed learning curve monotonic.
        with open(log_path) as f:
            lines = f.readlines()
        keep = [lines[0]]
        for ln in lines[1:]:
            try:
                if int(ln.split(",")[1]) <= start_steps:
                    keep.append(ln)
            except (IndexError, ValueError):
                pass
        with open(log_path, "w", newline="") as f:
            f.writelines(keep)
    with open(log_path, "a" if append_log else "w", newline="") as f:
        if not append_log:
            csv.writer(f).writerow([
                "update", "env_steps", "wall_s", "mean_reward",
                "mean_x_progress", "episodes", "policy_loss", "value_loss",
                "entropy", "fps"])

    total_steps = start_steps
    update = start_steps // steps_per_update
    start = time.perf_counter()
    if append_log:
        # keep the wall-clock column going as well instead of jumping to 0
        try:
            with open(log_path) as f:
                last = f.readlines()[-1].split(",")
            start -= float(last[2])
        except (OSError, IndexError, ValueError):
            pass
    episodes = 0
    ep_x = []

    base_update = start_steps // steps_per_update

    def anneal_lr(u):
        if args.lr_restart:
            # fresh fine-tuning round: run the schedule over the REMAINING
            # budget so the LR starts back at --lr instead of resuming at
            # whatever fraction the inherited step counter implies.
            denom = max(1, args.num_updates - base_update)
            frac = 1.0 - (u - base_update) / denom
        else:
            frac = 1.0 - u / args.num_updates
        frac = max(0.0, min(1.0, frac))
        for g in opt.param_groups:
            g["lr"] = max(args.lr * frac, 1e-5)

    while total_steps < args.timesteps:
        anneal_lr(update)
        t0 = time.perf_counter()

        # ---------------- collect ----------------
        if workers is not None:
            data = workers.collect(net.state_dict())
            for i, d in enumerate(data):
                buf.ram[:, i] = d["ram"]
                buf.img[:, i] = d["img"]
                buf.actions[:, i] = d["actions"]
                buf.logp[:, i] = d["logp"]
                buf.values[:, i] = d["values"]
                buf.rewards[:, i] = d["rewards"]
                buf.dones[:, i] = d["dones"]
                episodes += d["n_ep"]
                ep_x.extend(d["ep_x"])
            obs_np = collate([d["last_obs"] for d in data])
        else:
            for t in range(args.steps_per_update):
                obs_t = as_tensor(obs_np)
                with torch.no_grad():
                    a, logp, _, value = net.act(obs_t)
                a_np = a.cpu().numpy()
                buf.ram[t] = obs_np["ram"]
                buf.img[t] = obs_np["img"]
                buf.actions[t] = a_np
                buf.logp[t] = logp.cpu().numpy()
                buf.values[t] = value.cpu().numpy()

                next_obs = []
                for i, env in enumerate(envs):
                    o, r, done, info = env.step(int(a_np[i]))
                    if done:
                        episodes += 1
                        ep_x.append(info["x_progress"])
                        o = env.reset()
                    next_obs.append(o)
                    buf.rewards[t, i] = r
                    buf.dones[t, i] = float(done)
                obs_np = collate(next_obs)

        total_steps += steps_per_update
        collect_s = time.perf_counter() - t0

        with torch.no_grad():
            last_v = net.forward(as_tensor(obs_np))[1].numpy()

        adv, ret = compute_gae(buf.rewards, buf.dones, buf.values, last_v,
                               args.gamma, args.lam)

        # ---------------- update ----------------
        b_ram = torch.as_tensor(buf.ram.reshape(-1, OBS_DIM))
        b_img = torch.as_tensor(buf.img.reshape(-1, IMG_STACK, IMG_H, IMG_W))
        b_act = torch.as_tensor(buf.actions.reshape(-1))
        b_logp = torch.as_tensor(buf.logp.reshape(-1))
        b_adv = torch.as_tensor(adv.reshape(-1))
        b_ret = torch.as_tensor(ret.reshape(-1))

        flat = args.steps_per_update * args.nenvs
        idx = np.arange(flat)
        pl, vl, el = 0.0, 0.0, 0.0
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for start_i in range(0, flat, args.minibatch):
                mb = idx[start_i:start_i + args.minibatch]
                if len(mb) < 2:
                    continue
                logp, entropy, value = net.evaluate(
                    {"ram": b_ram[mb], "img": b_img[mb]}, b_act[mb])
                ratio = (logp - b_logp[mb]).exp()
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg = -(torch.min(ratio * mb_adv,
                                 ratio.clamp(1 - args.clip, 1 + args.clip) * mb_adv)).mean()
                vg = ((value - b_ret[mb]) ** 2).mean()
                loss = pg + args.vf_coef * vg - args.ent_coef * entropy.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                pl += pg.item(); vl += vg.item(); el += entropy.mean().item()

        update += 1
        n_mb = args.epochs * (flat // args.minibatch)
        mean_x = float(np.mean(ep_x[-50:])) if ep_x else 0.0
        fps = args.steps_per_update * args.nenvs / collect_s

        print(f"upd {update:4d} | steps {total_steps:8d} | "
              f"rew {buf.rewards.mean():7.3f} | x_prog {mean_x:8.1f} | "
              f"eps {episodes:4d} | pi {pl / n_mb:6.3f} | v {vl / n_mb:6.3f} | "
              f"ent {el / n_mb:5.3f} | {fps:5.0f} fps", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                update, total_steps, round(time.perf_counter() - start, 1),
                float(buf.rewards.mean()), mean_x, episodes,
                pl / n_mb, vl / n_mb, el / n_mb, round(fps, 1)])

        if update % args.save_every == 0:
            ck = os.path.join(args.save_dir, f"ppo_{total_steps:08d}.pt")
            torch.save({"model": net.state_dict(), "steps": total_steps,
                        "args": vars(args)}, ck)
            print(f"  saved {ck}")

    ck = os.path.join(args.save_dir, "ppo_final.pt")
    torch.save({"model": net.state_dict(), "steps": total_steps,
                "args": vars(args)}, ck)
    print(f"\ndone in {(time.perf_counter() - start) / 60:.1f} min -> {ck}")
    print(f"progress log: {log_path}")

    if workers is not None:
        workers.close()
    for e in envs:
        e.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--nenvs", type=int, default=8)
    p.add_argument("--workers", type=int, default=8,
                   help="worker PROCESSES, each with its own emulator and "
                        "policy copy.  0 = the old single-process loop "
                        "(8 serial environments).  Rules: one worker == one "
                        "environment, so --nenvs is overridden by this")
    p.add_argument("--steps-per-update", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=str, default="runs/ppo")
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start the network from an existing checkpoint")
    p.add_argument("--lr-restart", action="store_true",
                   help="run the LR schedule over the remaining budget so a "
                        "second fine-tuning round on inherited weights starts "
                        "at --lr instead of at the decayed fraction")
    p.add_argument("--resume", action="store_true",
                   help="continue the run in --save-dir: reload the newest "
                        "checkpoint and keep the step counter, the LR "
                        "schedule and the progress log going")
    p.add_argument("--no-start-axe", action="store_true",
                   help="do NOT arm the axe at reset: the agent has to walk "
                        "into the first egg itself (the real game).  Default "
                        "is to start every episode with the axe, which skips "
                        "the boring pre-axe phase and speeds up training")
    p.add_argument("--start-round", type=int, default=0,
                   help="0-based stage to start every episode in: 0 = 1-1 "
                        "(default), 1 = 1-2 (第二关), 2 = 1-3 ...  The boot "
                        "warps there by forcing $0038 while the stage is "
                        "built (tools/make_round2_state.py)")
    p.add_argument("--start-state", type=str, default=None,
                   help="load this cynes save state at reset instead of "
                        "booting (e.g. saves/round2_start_warp.state)")
    p.add_argument("--stages", type=str, default=None,
                   help="multi-stage curriculum: a weighted pool of stage "
                        "save states, e.g. '1-2:0.7,1-1:0.3'.  Each episode "
                        "draws one stage by weight (plan.md phase 3).  "
                        "Overrides --start-round/--start-state.")
    p.add_argument("--clear-ends-episode", action="store_true",
                   help="end the episode (and pay +5) the moment the flag "
                        "is cleared, instead of rolling into the tally")
    a = p.parse_args()

    # "1-2:0.7,1-1:0.3" -> ([paths], [weights])
    # Sorted into canonical stage order (1-1, 1-2, ... 8-4) so that the
    # one-hot index always means area*4 + round, no matter how the stages
    # were listed on the command line.  Before this the pool kept the CLI
    # order, so "1-2:0.7,1-1:0.3" made one-hot 0 mean 1-2 -- and every
    # playback tool that guesses the index from the file name (visualize.py)
    # fed the opposite bit to the policy.  (2026-10-17)
    a.stage_pool, a.stage_weights, stage_names = [], [], []
    if a.stages:
        for item in a.stages.split(","):
            item = item.strip()
            if not item:
                continue
            name, _, w = item.partition(":")
            name = name.strip()
            stage_names.append((name, float(w) if w else 1.0))
        stage_names.sort(key=lambda nw: tuple(int(p) for p in nw[0].split("-")))
        for name, w in stage_names:
            a.stage_pool.append(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "saves", "stages", f"stage_{name}.state"))
            a.stage_weights.append(w)
        for path in a.stage_pool:
            if not os.path.exists(path):
                p.error(f"no such stage save: {path}")
        print("stage pool: " + ", ".join(
            f"{n}={w:g}" for n, w in zip(
                [os.path.basename(p)[6:-6] for p in a.stage_pool],
                a.stage_weights)), flush=True)

    if a.start_round or a.start_state:
        print(f"stage start: round={a.start_round} state={a.start_state}",
              flush=True)
    if a.workers > 0:
        # one environment per worker process
        a.nenvs = a.workers
    a.num_updates = max(1, a.timesteps // (a.steps_per_update * a.nenvs))
    return a


if __name__ == "__main__":
    train(parse_args())
