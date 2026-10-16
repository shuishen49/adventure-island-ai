"""Does the agent actually KILL snails, or does it just walk into them?

Plays full episodes with the trained policy and, for every decision step,
records what is happening around the player:

  * nearest enemy distance (dx, in px, signed: + = to the right)
  * whether an axe (object id 66) is in the air
  * whether the chosen action throws
  * the kill counter of the env (enemy slot vanished near us while an axe
    was airborne)

From that we get the numbers that matter:

  kills per episode      - is it killing anything at all?
  throws per episode     - how often does it use the axe?
  kill / throw ratio     - is throwing aimed, or just spam?
  distance at throw      - does it throw early enough?  An axe leaves the
                           player and needs a few frames to travel, so
                           throwing at dx < ~20 is already too late.
  what it died to        - "hugged an enemy" (|dx| < 24 at death, i.e. it
                           walked into it) vs falling vs starving.

    python tools/kill_or_crash.py <checkpoint> [episodes]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES  # noqa: E402
import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, ROM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic  # noqa: E402


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_act = int(ck["model"]["policy.weight"].shape[0])
    net = ActorCritic(OBS_DIM, n_act, 128)
    net.load_state_dict(ck["model"], strict=False)
    net.eval()
    return net, n_act, int(ck.get("steps", 0))


def nearest_enemy(nes) -> tuple[float, int, int]:
    """(dx of nearest enemy, n enemies on screen, n axes in the air)."""
    p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
    best, n_enemy, n_shot = 9999.0, 0, 0
    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        if oid == 0:
            continue
        dx = R.px(R.read_obj_world_x(nes, s) - p_x)
        if oid in R.ENEMY_IDS:
            n_enemy += 1
            if abs(dx) < abs(best):
                best = dx
        elif oid == R.ID_PROJECTILE:
            n_shot += 1
    return (best if n_enemy else 9999.0), n_enemy, n_shot


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_v6/ppo_00020480.pt"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    # default: SAMPLE, exactly like training does (greedy tends to walk a
    # single deterministic path straight into the first snail)
    greedy = "--greedy" in sys.argv

    net, n_act, steps = load(ckpt)
    mode = "greedy" if greedy else "采样(同训练)"
    print(f"checkpoint {ckpt} ({steps} steps, {n_act} actions), "
          f"{n_ep} episodes, {mode}\n", flush=True)

    env = AdventureIslandEnv(seed=1234, start_with_axe=True)
    nes = env._nes          # read-only peek, we never write
    rng = np.random.default_rng(0)

    throw_dist: list[float] = []      # nearest enemy dx when throwing
    kill_dist: list[float] = []       # dx of the enemy that died
    deaths: list[tuple[str, float, bool, bool]] = []   # cause, dx, threw, axe
    tot_kills = tot_throws = tot_deaths = 0
    axe_steps = shot_steps = 0
    throw_armed: list[bool] = []
    steps_total = 0
    ep_rows = []

    for ep in range(n_ep):
        obs = env.reset(seed=ep)
        kills = throws = my_deaths = 0
        hist: list[tuple[float, int, str]] = []   # (dx, n_shot, action)
        while True:
            with torch.no_grad():
                t = {"ram": torch.as_tensor(obs["ram"], dtype=torch.float32)
                     .unsqueeze(0),
                     "img": torch.as_tensor(obs["img"], dtype=torch.uint8)
                     .unsqueeze(0)}
                logits = net(t)[0].detach().numpy().ravel()
                if greedy:
                    action = int(logits.argmax())
                else:
                    p = np.exp(logits - logits.max())
                    action = int(rng.choice(len(logits), p=p / p.sum()))
            name = ACTIONS[action][1]
            dx, n_enemy, n_shot = nearest_enemy(nes)
            armed = nes[R.WEAPON] > 0
            axe_steps += int(armed)
            if n_shot > 0:
                shot_steps += 1
            if "throw" in name:
                throws += 1
                throw_armed.append(armed)
                if n_enemy:
                    throw_dist.append(dx)

            obs, rew, done, info = env.step(action)
            kills += int(info.get("kills", 0))
            steps_total += 1
            if info.get("kills", 0):
                kill_dist.append(dx)
            hist.append((dx, n_shot, name))

            if done:
                if info.get("death"):
                    my_deaths += 1
                    # look back a few decisions: did it throw at all, and
                    # was an axe already flying?
                    last = hist[-6:]
                    threw = any("throw" in h[2] for h in last)
                    axe_up = any(h[1] > 0 for h in last)
                    d_dx = last[-1][0]
                    if abs(d_dx) < 24 and last[-1][1] == 0:
                        cause = "撞上敌人"
                    elif abs(d_dx) < 24:
                        cause = "撞上敌人(斧头已出手)"
                    else:
                        cause = "掉坑/其他"
                    deaths.append((cause, d_dx, threw, axe_up))
                break

        tot_kills += kills
        tot_throws += throws
        tot_deaths += my_deaths
        ep_rows.append((ep, env._steps, info["x_progress"], kills, throws,
                        my_deaths, info.get("score", 0)))
        print(f"  ep{ep}: {env._steps:4d} 步  x={info['x_progress']:7.0f}px  "
              f"击杀 {kills}  扔斧 {throws}  死亡 {my_deaths}  "
              f"分数 {info.get('score', 0)}", flush=True)

    env.close()
    del nes

    print(f"\n合计: 击杀 {tot_kills}   扔斧 {tot_throws}   "
          f"死亡 {tot_deaths}   平均击杀/局 {tot_kills / n_ep:.2f}")
    print(f"持有斧头的步数: {axe_steps}/{steps_total} "
          f"({axe_steps / max(steps_total, 1):.0%})   "
          f"空中有斧头(SHOT66)的步数: {shot_steps} "
          f"({shot_steps / max(steps_total, 1):.0%})")
    if throw_armed:
        print(f"扔斧时手里有武器: {sum(throw_armed)}/{len(throw_armed)} "
              f"({np.mean(throw_armed):.0%})  "
              f"← 低说明它是\"空手按B\"，白按")
    if tot_throws:
        print(f"命中率 (击杀/扔斧): {tot_kills / tot_throws:.2f}")
    if throw_dist:
        a = np.array(throw_dist)
        print(f"扔斧时敌人距离: 中位 {np.median(a):6.0f}px  "
              f"均值 {a.mean():6.0f}px  "
              f"过近(<24px)占比 {(np.abs(a) < 24).mean():.0%}")
    else:
        print("扔斧时附近没有敌人（纯空放）")
    if kill_dist:
        a = np.array(kill_dist)
        print(f"击杀时敌人距离: 中位 {np.median(a):6.0f}px  "
              f"均值 {a.mean():6.0f}px")

    print("\n死亡分析:")
    if not deaths:
        print("  没有死亡")
    for cause, dx, threw, axe_up in deaths:
        print(f"  {cause:<18} 最近敌人 {dx:+6.0f}px  "
              f"死前扔过斧 {'是' if threw else '否'}  "
              f"空中有斧头 {'是' if axe_up else '否'}")


if __name__ == "__main__":
    main()
