"""What actually kills the agent?

The user's correction: a campfire is LETHAL (you lose a life), it is not
the harmless "costs some stamina" obstacle.  So the question is: at the
exact moment a life is lost, which object id is nearest?

We snapshot every step and report the PREVIOUS snapshot when the life
counter drops (by the time `done` comes back the game is already in its
death animation / respawn).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic, newest_checkpoint  # noqa: E402
from visualize import objects  # noqa: E402

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CKPT = sys.argv[2] if len(sys.argv) > 2 else (
    newest_checkpoint("runs/ppo_v9") or "runs/ppo/ppo_final.pt")


def snapshot(nes):
    objs = objects(nes)
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px, py = (player[2], player[3]) if player else (0, 0)
    out = []
    for s, oid, x, y, vx in objs:
        if s == R.PLAYER_SLOT:
            continue
        out.append((abs(x - px), oid, x - px, y - py, vx, s))
    out.sort()
    return out, px, py


def main():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"], strict=False)
    net.eval()
    print(f"driven by {CKPT} ({ck['steps']} steps), {N_EP} episodes")

    env = AdventureIslandEnv(seed=0, start_with_axe=True)
    nes = env._nes
    causes: dict[int, int] = {}
    detail: list[tuple[int, int, float, float, int]] = []
    ep_end: dict[str, int] = {}
    deaths = 0

    for ep in range(N_EP):
        obs = env.reset(seed=ep)
        lives = nes[R.LIVES]
        prev_snap = snapshot(nes)[0]
        while True:
            with torch.no_grad():
                t = {"ram": torch.as_tensor(obs["ram"],
                                            dtype=torch.float32).unsqueeze(0),
                     "img": torch.as_tensor(obs["img"],
                                            dtype=torch.uint8).unsqueeze(0)}
                a = int(net(t)[0].argmax())
            obs, r, done, info = env.step(a)
            snap = snapshot(nes)[0]
            # `info["death"]` is the env's own verdict (lives < start_lives,
            # or the dying flag).  The local `nes[LIVES] < lives` comparison
            # below used to be the only test and it silently under-reported:
            # keep both and print the disagreement so the two never drift
            # apart again.
            flag_dead = bool(info.get("death"))
            ram_dead = nes[R.LIVES] < lives
            if flag_dead or ram_dead:
                deaths += 1
                near = prev_snap[0] if prev_snap else None
                if near:
                    dist, oid, dx, dy, vx, slot = near
                    causes[oid] = causes.get(oid, 0) + 1
                    detail.append((oid, slot, dx, dy, vx))
                    print(f"  ep{ep} death #{deaths}: id{oid} "
                          f"({R.kind_of(oid)[0]}) at dx={dx:+.0f} "
                          f"dy={dy:+.0f} vx={vx}  slot{slot}", flush=True)
                else:
                    causes[-1] = causes.get(-1, 0) + 1
                    print(f"  ep{ep} death #{deaths}: no object nearby "
                          f"(fell in a pit?)", flush=True)
                if flag_dead and not ram_dead:
                    print(f"      (env flagged death but RAM lives "
                          f"{nes[R.LIVES]} >= {lives})")
                lives = nes[R.LIVES]
            else:
                if ram_dead and not flag_dead:
                    pass
                lives = nes[R.LIVES]
            prev_snap = snap
            if done:
                if not flag_dead:
                    ep_end[info.get("end", "?")] = \
                        ep_end.get(info.get("end", "?"), 0) + 1
                break
    env.close()

    print(f"\n{deaths} deaths.  nearest object at the moment of death:")
    for oid, n in sorted(causes.items(), key=lambda kv: -kv[1]):
        name = "fell / unknown" if oid == -1 else R.kind_of(oid)[0]
        print(f"   id{oid:<4} {name:<10} {n:3d}  ({n / max(1, deaths):.0%})")
    if ep_end:
        print(f"episodes that ended WITHOUT the death flag: {ep_end}")


if __name__ == "__main__":
    main()
