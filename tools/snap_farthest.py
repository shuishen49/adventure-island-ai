"""Play the newest checkpoint and keep the FARTHEST frame it ever reached.

Runs through the very same env the trainer uses (same reset, same random
warm-up, same observation), so the distance matches what the progress log
reports.  The point is to look at the overlay out there, where the level
gets hard: is the campfire / obstacle marked, and which ids actually spawn.

    python tools/snap_farthest.py runs/ppo_v9/ppo_00614400.pt 6

writes outputs/farthest.png (annotated, 3x) and prints the object list of
that exact frame.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import ram_map as R  # noqa: E402
from env import ACTIONS, OBS_DIM, AdventureIslandEnv  # noqa: E402
from train import ActorCritic, newest_checkpoint  # noqa: E402
from visualize import annotate, objects  # noqa: E402

OUT = "outputs/farthest.png"
OUT_RAW = "outputs/farthest_raw.png"
OUT_FIRE = "outputs/campfire.png"


def object_table(nes) -> list[tuple[int, int, str, float, float, int, int]]:
    """(slot, id, label, dx, dy, screen_x, screen_y) for on-screen slots.

    Uses the SAME screen coordinates the overlay draws with, so this list
    matches the boxes in the picture exactly.  Off-screen slots and stale
    slots whose world x wrapped into nonsense (dx = +65430) are skipped.
    """
    objs = objects(nes)
    player = next((o for o in objs if o[0] == R.PLAYER_SLOT), None)
    px_x, px_y = (player[2], player[3]) if player else (0, 0)
    rows = []
    for s, oid, x, y, vx in objs:
        if not (-8 <= x < 264 and -8 <= y < 248):
            continue
        rows.append((s, oid, R.kind_of(oid)[0], x - px_x, y - px_y, x, y))
    return rows


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else ""
    if not ckpt:
        ckpt = newest_checkpoint("runs/ppo_v9") or "runs/ppo/ppo_final.pt"
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    missing, _ = net.load_state_dict(ck["model"], strict=False)
    net.eval()
    print(f"checkpoint {ckpt} ({ck['steps']} steps)"
          f"{'  MISSING: ' + str(missing) if missing else ''}")

    env = AdventureIslandEnv(seed=0, start_with_axe=True)
    tracker = getattr(env, "_activity", None)

    best_x, best_img, best_rows, best_raw = -1.0, None, None, None
    # closest sighting of a ground obstacle (id 2 = campfire)
    fire_d, fire_img, fire_rows = 1e9, None, None
    best_ep, best_act, xs = -1, "", []

    for ep in range(n_ep):
        obs = env.reset(seed=ep)
        if tracker is not None:
            tracker.reset()
        action = 0
        while True:
            with torch.no_grad():
                t = {"ram": torch.as_tensor(obs["ram"],
                                            dtype=torch.float32).unsqueeze(0),
                     "img": torch.as_tensor(obs["img"],
                                            dtype=torch.uint8).unsqueeze(0)}
                action = int(net(t)[0].argmax())
            obs, r, done, info = env.step(action)
            x = info["x_progress"]
            if x > best_x:
                best_x = x
                best_raw = env._last_frame.copy()
                best_img = annotate(env._last_frame.copy(), env._nes,
                                    ACTIONS[action][1], tracker)
                best_rows = object_table(env._nes)
                best_ep, best_act = ep, ACTIONS[action][1]
            for _s, _oid, _lab, _dx, _dy, _sx, _sy in (
                    object_table(env._nes)):
                if (_oid in (2, 1, 19, 25, 48) and 20 < _dx < 220
                        and abs(_dy) < 30 and _dx < fire_d):
                    fire_d = _dx
                    fire_img = annotate(env._last_frame.copy(),
                                        env._nes,
                                        ACTIONS[action][1],
                                        tracker)
                    fire_rows = object_table(env._nes)
            if done:
                break
        xs.append(x)
        print(f"  ep{ep}: {x:7.0f} px   (farthest so far {best_x:.0f})",
              flush=True)
    env.close()

    print(f"\nfarthest point: {best_x:.0f} px  (ep{best_ep}, action "
          f"{best_act})   per-episode: {[round(v) for v in xs]}")
    print("objects on that frame (dx/dy in px, measured from the player; "
          "sx/sy = screen pixel, same boxes as in the picture):")
    print("  slot   id  label       dx     dy     sx    sy")
    for s, oid, label, dx, dy, sx, sy in best_rows:
        print(f"  {s:4d} {oid:4d}  {label:<8} {dx:+6.0f} {dy:+6.0f} "
              f"{sx:5.0f} {sy:5.0f}")

    if fire_img is not None:
        print(f"\nclosest ground obstacle (campfire candidate): "
              f"{fire_d:.0f} px ahead")
        print("  slot   id  label       dx     dy     sx    sy")
        for s, oid, label, dx, dy, sx, sy in fire_rows:
            print(f"  {s:4d} {oid:4d}  {label:<8} {dx:+6.0f} {dy:+6.0f} "
                  f"{sx:5.0f} {sy:5.0f}")
        fi = Image.fromarray(fire_img)
        fi.resize((fi.width * 3, fi.height * 3), Image.NEAREST).save(OUT_FIRE)
        print(f"wrote {OUT_FIRE}")

    img = Image.fromarray(best_img)
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(OUT)
    Image.fromarray(best_raw).resize(
        (best_raw.shape[1] * 3, best_raw.shape[0] * 3),
        Image.NEAREST).save(OUT_RAW)
    print(f"\nwrote {OUT}  (+ {OUT_RAW} without the overlay)")


if __name__ == "__main__":
    main()
