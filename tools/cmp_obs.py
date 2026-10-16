"""Why does the agent never throw inside the training env?

The window (enjoy.build_obs) shows ~33 % throwing, but the same weights
inside AdventureIslandEnv throw 0 times and die on the first snail.  So
one of the two observation paths is lying.  Per decision step this prints

  truth     - enemies actually present, read straight from RAM
  seen      - what the env TELLS the agent (e_dx / e_seen in obs["ram"])
  zombie    - slots the zombie filter is hiding
  act(env)  - greedy action with the env observation
  act(raw)  - greedy action with the same observation built by
              enjoy.build_obs (no zombie filter)

If `truth` says a snail is 40 px away while `seen` says e_seen = 0, the
zombie filter is blinding the agent and that is why it never throws.

    python tools/cmp_obs.py <checkpoint> [steps]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES  # noqa: E402
import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic  # noqa: E402


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n = int(ck["model"]["policy.weight"].shape[0])
    net = ActorCritic(OBS_DIM, n, 128)
    net.load_state_dict(ck["model"], strict=False)
    net.eval()
    return net, n


def truth_enemies(nes) -> list[tuple[int, float]]:
    p_x = R.read_obj_world_x(nes, R.PLAYER_SLOT)
    out = []
    for s in range(R.N_OBJ_SLOTS):
        oid = nes[R.OBJ.id + R.OBJ.STRIDE * s]
        if oid in R.ENEMY_IDS:
            out.append((s, R.px(R.read_obj_world_x(nes, s) - p_x)))
    return out


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/ppo_v6/ppo_00020480.pt"
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    net, n_act = load(ckpt)
    env = AdventureIslandEnv(seed=0, start_with_axe=True)
    nes = env._nes
    obs = env.reset(seed=0)

    print(f"{ckpt} ({n_act} actions)\n")
    print(" step | truth(slot@dx)        | seen  e_dx  | zombie | "
          "act(env)            | act(shim)")
    mism = 0
    for i in range(n_steps):
        truth = truth_enemies(nes)
        ram = obs["ram"]
        e_dx, e_seen = ram[-9] * 256.0, ram[-7]
        zombies = [s for s, _ in truth if env._is_zombie(s, 14)]

        with torch.no_grad():
            t_env = {"ram": torch.as_tensor(ram, dtype=torch.float32)
                     .unsqueeze(0),
                     "img": torch.as_tensor(obs["img"], dtype=torch.uint8)
                     .unsqueeze(0)}
            a_env = int(net(t_env)[0].argmax())

        # the very same features, but WITHOUT the zombie filter
        import enjoy
        o2 = enjoy.build_obs(nes)
        with torch.no_grad():
            t2 = {"ram": torch.as_tensor(o2["ram"], dtype=torch.float32)
                  .unsqueeze(0),
                  "img": torch.as_tensor(obs["img"], dtype=torch.uint8)
                  .unsqueeze(0)}
            a2 = int(net(t2)[0].argmax())

        near = [f"{s}@{dx:+.0f}" for s, dx in truth if abs(dx) < 400]
        with torch.no_grad():
            logits = net(t_env)[0].numpy().ravel()
        top = np.argsort(-logits)[:3]
        tops = " ".join(f"{ACTIONS[j][1]}={logits[j]:+.2f}" for j in top)
        print(f"{i:5d} | {','.join(near) or '-':<20} | "
              f"{e_seen:4.1f} {e_dx:+6.0f} | {str(zombies):<6} | "
              f"{ACTIONS[a_env][1]:<18} | {tops}")
        if a_env != a2:
            mism += 1

        obs, rew, done, info = env.step(a_env)
        if done:
            print(f"  -> died at step {i} (x={info['x_progress']:.0f}px, "
                  f"kills={info.get('kills', 0)})")
            break

    env.close()
    del nes
    print(f"\nenv 观测与 shim 观测给出不同动作: {mism}/{n_steps} 步")


if __name__ == "__main__":
    main()
