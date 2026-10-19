"""Pin down the boss HP byte by counting the on-screen HP bar in PIXELS.

The bar is drawn with background tiles (it is not in the OAM shadow at
$0200 at all), so counting sprites fails.  But it is a row of bright blocks
on the dark HUD strip at the top of the frame, so we can just read pixels.

Plan: teleport the player right in front of the boss so axes actually
connect, log the pixel-counted bar length plus every RAM byte each
decision, then diff the RAM exactly at the decisions where the bar shrank.

    python tools/probe_boss_v3.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (ACTIONS, AdventureIslandEnv, N_ACTIONS,   # noqa: E402
                 OBS_DIM)
from train import ActorCritic                              # noqa: E402

CKPT = os.path.join(_ROOT, "models", "ppo_1-3_clear_16M.pt")
OID, ST = R.OBJ.id, R.OBJ.STRIDE
A_RT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+throw"][0]
A_RJT = [i for i, a in enumerate(ACTIONS) if a[1] == "right+jump+throw"][0]
A_RIGHT = [i for i, a in enumerate(ACTIONS) if a[1] == "right"][0]
A_NOOP = [i for i, a in enumerate(ACTIONS) if a[1] == "noop"][0]


def bar_len(frame: np.ndarray) -> int:
    """count bright blocks on the HUD strip (rows 38..44, x 75..145)."""
    strip = frame[36:46].astype(np.int16).mean(axis=2)     # H, W grey
    bright = strip > 120
    cols = bright[:, 75:150].any(axis=0)
    runs, in_run = 0, False
    for c in cols:
        if c and not in_run:
            runs += 1
            in_run = True
        elif not c:
            in_run = False
    return runs


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()

    env = AdventureIslandEnv(start_round=3, start_with_axe=True,
                             stage_index=3)
    env.reset(seed=0)
    nes = env._nes
    o = env._obs()
    action, frame, best, stalls = 0, 0, 0.0, 0
    while frame < 60 * 60 * 4:
        if frame % env.frame_skip == 0:
            with torch.no_grad():
                lg, _ = net({"img": torch.as_tensor(o["img"][None]),
                             "ram": torch.as_tensor(o["ram"][None],
                                                    dtype=torch.float32)})
            action = int(lg[0].argmax())
        nes.controller = ACTIONS[action][0]
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        arr = np.asarray(nes.step(), dtype=np.uint8)
        nes.controller = 0
        if frame % env.frame_skip == env.frame_skip - 1:
            env._push_frame(arr)
        o = env._obs()
        frame += 1
        p = env._cam.progress(nes)
        if p > best + 0.5:
            best, stalls = p, 0
        else:
            stalls += 1
        if stalls > 45 * 60:
            break
    print(f"arena: progress={best:.0f}", flush=True)
    print("bar length at entry:", bar_len(arr))

    # --- teleport the player in front of the boss ----------------------
    boss_slot = None
    for s in range(1, 16):
        oid = int(nes[OID + ST * s])
        x = R.px(R.read_obj_world_x(nes, s))
        if oid and 200 < x < 260:
            boss_slot = s
            break
    print("boss slot:", boss_slot,
          "" if boss_slot is None else
          f"id={int(nes[OID + ST * boss_slot])}")
    if boss_slot is not None:
        for off in (R.OBJ.x_hi, R.OBJ.x_lo, R.OBJ.x_fr):
            nes[off + ST * 0] = nes[off + ST * boss_slot]
        nes[R.OBJ.x_lo + ST * 0] = (int(nes[R.OBJ.x_lo + ST * 0]) - 24) & 0xFF

    # --- spam axes, log bar length + full RAM --------------------------
    bars, rams = [], []
    for t in range(1200):
        nes[R.DEATH] = 0
        nes[R.STAMINA] = 10
        nes[R.WEAPON] = 1
        # boss hovers ABOVE the axe's ground-hugging flight path: only a
        # jump-throw has any chance of connecting
        act = A_RJT if t % 8 in (0, 1, 2) else A_RIGHT
        env.step(act)
        if t % 40 == 0:
            bx = R.px(R.read_obj_world_x(nes, 14))
            by = R.px(R.read_obj_world_y(nes, 14))
            print(f"  t={t:4d} bar={bar_len(np.asarray(nes.step())):2d}"
                  f"  $0011={int(nes[0x11]):3d}  boss_slot14 id={int(nes[OID + ST * 14])}"
                  f" x={bx:6.1f} y={by:6.1f}", flush=True)
        f = np.asarray(nes.step(), dtype=np.uint8).copy()
        bars.append(bar_len(f))
        rams.append(np.array([nes[a] for a in range(2048)], dtype=np.uint8))
    bars = np.array(bars)
    rams = np.array(rams)
    print("bar length series:", bars.tolist())
    drops = np.where(np.diff(bars) < 0)[0]
    print("bar dropped at decisions:", drops.tolist())
    if len(drops) == 0:
        print("!! bar never shrank -- axes are not connecting")
        env.close()
        return

    # at each drop, which RAM bytes went DOWN by 1 in that window?
    hits: dict[int, int] = {}
    for d in drops:
        lo, hi = max(0, d - 3), min(len(bars) - 1, d + 3)
        for a in range(2048):
            w = rams[lo:hi + 1, a].astype(int)
            if w[0] - w[-1] == 1 and (np.diff(w) <= 0).sum() >= 1 \
                    and (np.diff(w) > 0).sum() == 0:
                hits[a] = hits.get(a, 0) + 1
    print("bytes that fell by 1 in EVERY drop window:")
    for a, n in sorted(hits.items(), key=lambda kv: -kv[1]):
        if n == len(drops):
            print(f"  ${a:04X}  (all {n}/{len(drops)} drops)  "
                  f"start={rams[0, a]} end={rams[-1, a]}")
    env.close()


if __name__ == "__main__":
    main()
