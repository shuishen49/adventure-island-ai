"""What is object id 4?

It shows up glued to the player in the farthest frame (dx=+4, dy=+6) and
the agent keeps dying right there, yet it is only tagged "ENEMY?" (the
doubtful entry that came out of the flawed stamina heuristic).  A campfire
would be: static (vx = 0), sitting on the ground, and - unlike a snail -
it should NOT walk towards us.

For every id-4 sighting this prints its velocity, height, how it moves
relative to the player, and what happens to lives / stamina on contact.
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

N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
CKPT = sys.argv[2] if len(sys.argv) > 2 else (
    newest_checkpoint("runs/ppo_v9") or "runs/ppo/ppo_final.pt")


def signed(b):
    return b - 256 if b > 127 else b


def main():
    # Random play never gets far enough to meet id 4 (it only shows up in
    # the deeper part of the stage), so the AGENT drives this survey.
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"], strict=False)
    net.eval()
    print(f"driven by {CKPT} ({ck['steps']} steps), {N_EP} episodes")

    env = AdventureIslandEnv(seed=0, start_with_axe=True)
    nes = env._nes

    print(" id  n_see  vx:mode  dy:med  walking?  contact:lives/stamina")
    stats: dict[int, dict] = {}
    contacts: dict[int, list] = {}

    for ep in range(N_EP):
        obs = env.reset(seed=ep)
        prev = {}
        while True:
            with torch.no_grad():
                t = {"ram": torch.as_tensor(obs["ram"],
                                            dtype=torch.float32).unsqueeze(0),
                     "img": torch.as_tensor(obs["img"],
                                            dtype=torch.uint8).unsqueeze(0)}
                a = int(net(t)[0].argmax())
            obs, r, done, info = env.step(a)
            cur = {}
            for s, oid, x, y, vx in objects(nes):
                if s == R.PLAYER_SLOT:
                    continue
                cur[s] = (oid, x, y, vx)
            for s, (oid, x, y, vx) in cur.items():
                st = stats.setdefault(
                    oid, {"n": 0, "vx": [], "dy": [], "walk": 0})
                st["n"] += 1
                st["vx"].append(vx)
                st["dy"].append(y - 153)          # 153 = player ground y
                # does it close in on us over consecutive sightings?
                if s in prev and prev[s][0] == oid:
                    if abs(x - prev[s][1]) > 0.6:
                        st["walk"] += 1
            # contact: something was within 12 px and is gone/damaging now
            for s, (oid, x, y, vx) in cur.items():
                if abs(x - 97) < 14 and abs(y - 153) < 40:
                    contacts.setdefault(oid, []).append(
                        (info["lives"], info["stamina"], info["death"]))
            prev = cur
            if done:
                break
    env.close()

    rows = sorted(stats.items(), key=lambda kv: -kv[1]["n"])[:14]
    print(f"{'id':>4} {'seen':>6} {'vx':>7} {'dy':>7} {'moved':>7}  contacts")
    for oid, st in rows:
        vx = np.array(st["vx"])
        # mode of the velocity: 0 == static obstacle
        vals, cnts = np.unique(vx, return_counts=True)
        mode = int(vals[cnts.argmax()])
        dy = float(np.median(st["dy"]))
        moved = st["walk"] / max(1, st["n"])
        cc = contacts.get(oid, [])
        if cc:
            deaths = sum(1 for lv, sta, d in cc if d)
            c_txt = (f"{len(cc):4d} frames  death={deaths}  "
                     f"sta {min(s for _, s, _ in cc)}-"
                     f"{max(s for _, s, _ in cc)}")
        else:
            c_txt = "-"
        print(f"{oid:4d} {st['n']:6d} {mode:7d} {dy:7.0f} {moved:7.0%}  {c_txt}")


if __name__ == "__main__":
    main()
