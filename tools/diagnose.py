"""Post-mortem for a trained policy: WHY does it die, and what does it do?

For every episode we record
  * how far it got, how long it lived
  * the cause of death: fell in a pit / starved (stamina) / hit something
  * the action histogram (does it ever throw the axe? does it jump?)
  * how often it picked up the axe

Usage:  python tools/diagnose.py [checkpoint] [episodes]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cynes import NES, NES_INPUT_START  # noqa: E402
import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, ROM  # noqa: E402
from enjoy import build_obs, push_img, reset_cam  # noqa: E402
from train import ActorCritic  # noqa: E402
from visualize import boot  # noqa: E402

GROUND_Y = 200.0        # player y below this = still on/near a platform


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    # older checkpoints have an 8-action head; the first 8 entries of
    # ACTIONS are unchanged, so their indices still mean the same thing
    n_actions = ck["model"]["policy.weight"].shape[0]
    net = ActorCritic(OBS_DIM, n_actions, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck["steps"], n_actions


def what_killed(nes, p_y, near_enemy, sta_before) -> str:
    if p_y > GROUND_Y:
        return "pit"
    if sta_before <= 1:
        return "starve"
    if near_enemy:
        return "enemy"
    return "other"


def run(net, nes, max_frames: int = 4500, stochastic: bool = False,
        warmup: int = 0, rng: np.random.Generator | None = None,
        n_actions: int | None = None) -> dict:
    """Play one episode.

    `warmup` frames of random input first: the game is completely
    deterministic, so without it every "episode" is byte-identical and
    averaging over N of them is meaningless.
    """
    nes.reset() if hasattr(nes, "reset") else None
    boot(nes)
    reset_cam(nes)

    start_lives = nes[R.LIVES]
    last_lives = start_lives
    last_weapon = nes[R.WEAPON]
    last_sta = nes[R.STAMINA]
    max_x = 0.0
    axes = 0
    deaths = []
    acts = np.zeros(len(ACTIONS), dtype=int)
    sta_hist = []

    cam = R.CameraTracker(nes)
    for i in range(max_frames):
        if i < warmup and rng is not None:
            action = int(rng.integers(n_actions or len(ACTIONS)))
        elif i % 4 == 0:
            obs = build_obs(nes)
            with torch.no_grad():
                t = {"ram": torch.as_tensor(obs["ram"], dtype=torch.float32)
                     .unsqueeze(0),
                     "img": torch.as_tensor(obs["img"], dtype=torch.uint8)
                     .unsqueeze(0)}
                logits = net(t)[0]
                if stochastic:
                    action = int(torch.distributions.Categorical(
                        logits=logits).sample())
                else:
                    action = int(logits.argmax())
        acts[action] += 1
        nes.controller = ACTIONS[action][0]
        push_img(nes, np.asarray(nes.step(), dtype=np.uint8))

        p_y = R.px(R.read_obj_world_y(nes, R.PLAYER_SLOT))
        max_x = max(max_x, cam.progress(nes))
        sta = nes[R.STAMINA]
        sta_hist.append(sta)

        # nearest enemy distance (screen px)
        p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
        near = False
        for s in range(1, 15):
            oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
            if oid not in R.ENEMY_IDS:
                continue
            if abs(R.px(R.read_obj_world_x(nes, s) - p_x)) < 12 and \
                    abs(R.px(R.read_obj_world_y(nes, s)) - R.px(
                        R.read_obj_world_y(nes, R.PLAYER_SLOT))) < 16:
                near = True

        lv = nes[R.LIVES]
        if lv < last_lives:
            deaths.append(what_killed(nes, p_y, near, last_sta))
            last_lives = lv
        w = nes[R.WEAPON]
        if w > last_weapon:
            axes += 1
        last_weapon = w
        last_sta = sta
        if lv == 0:
            break
    nes.controller = 0
    return {"x": max_x, "frames": i + 1, "deaths": deaths, "acts": acts,
            "axes": axes, "min_sta": min(sta_hist) if sta_hist else 0,
            "mean_sta": float(np.mean(sta_hist)) if sta_hist else 0}


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo/ppo_final.pt"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    warmup = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    net, steps, n_act = load(ckpt)
    print(f"checkpoint {ckpt} ({steps} steps), {n_ep} episodes, "
          f"warmup={warmup} random frames\n")
    nes = NES(ROM)

    xs, all_deaths = [], []
    acts = np.zeros(len(ACTIONS), dtype=int)
    axes = 0
    rng = np.random.default_rng(0)
    for e in range(n_ep):
        r = run(net, nes, stochastic=True, warmup=warmup, rng=rng,
                  n_actions=n_act)
        xs.append(r["x"])
        all_deaths += r["deaths"]
        acts += r["acts"]
        axes += r["axes"]
        print(f"  ep{e+1:2d}: {r['x']:7.0f} px  {r['frames']:4d} 帧  "
              f"死亡 {len(r['deaths'])} ({','.join(r['deaths']) or '-'})  "
              f"斧头 {r['axes']}  最低体力 {r['min_sta']}")
    del nes

    xs = np.array(xs)
    print(f"\n平均 {xs.mean():.0f} px   中位 {np.median(xs):.0f}   "
          f"最好 {xs.max():.0f}   最差 {xs.min():.0f}")
    from collections import Counter
    print("死因统计:", dict(Counter(all_deaths)))
    print("总死亡次数:", len(all_deaths), f" 平均每局 {len(all_deaths)/n_ep:.2f}")
    print(f"拿斧头: 共 {axes} 次, 平均每局 {axes/n_ep:.2f}")
    print("动作分布:")
    tot = acts.sum()
    for i, (_, name) in enumerate(ACTIONS):
        print(f"   {name:<12} {acts[i]:6d}  {acts[i]/tot*100:5.1f}%")


if __name__ == "__main__":
    main()
