"""Two questions the user raised, answered with measurements.

1. "the观察窗 has no health bar" -- when the boss (id 67 body / 198 head)
   is on screen, does the HUD change?  Compare the top HUD strip pixel-by
   pixel between normal play and the boss fight.

2. "hitting the HEAD works" -- then the boss must have an HP byte.  Find it
   the only honest way: same snapshot, two branches
       A: stand still        (nothing should damage the boss)
       B: jump + throw at the head
   and diff the whole 2 KB of RAM.  A byte that only ever goes DOWN in B
   and never moves in A is the HP byte.

Starts from a safe state right in front of the boss (no cheats).

    python tools/probe_boss_fight.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import ram_map as R                                        # noqa: E402
from env import (AdventureIslandEnv, NES_INPUT_RIGHT,   # noqa: E402
                 NES_INPUT_A, NES_INPUT_B)
from cynes import NES                                      # noqa: E402

N_RAM = 2048
BOSS = (67, 198)


def slots(nes):
    out = []
    for s in range(1, 16):
        o = R.OBJ.STRIDE * s
        v = int(nes[R.OBJ.id + o])
        if v:
            out.append((s, v,
                        R.px(R.read_obj_world_x(nes, s)),
                        R.px(R.read_obj_world_y(nes, s))))
    return out


def has_boss(nes) -> bool:
    return any(int(nes[R.OBJ.id + R.OBJ.STRIDE * s]) in BOSS
               for s in range(1, 16))


def hud_strip(frame: np.ndarray, y0=8, y1=48) -> np.ndarray:
    """Top HUD strip as grey (the lives / stamina / score bar lives here)."""
    g = frame.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    return g[y0:y1, :]


def hud_stats(strip: np.ndarray) -> tuple[int, float]:
    """(non-background pixels, mean brightness) -- a crude HUD meter."""
    nb = int((strip > 40).sum())
    return nb, float(strip.mean())


def ram(nes) -> np.ndarray:
    return np.frombuffer(bytes(nes[a] for a in range(N_RAM)),
                         dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=os.path.join(
        _ROOT, "saves", "stages", "stage_1-4_safe5400.state"))
    ap.add_argument("--frames", type=int, default=2500)
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs",
                                                  "bossfight"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch                                           # noqa: E402
    from env import ACTIONS, N_ACTIONS, OBS_DIM            # noqa: E402
    from train import ActorCritic                          # noqa: E402
    env = AdventureIslandEnv(start_with_axe=True, start_round=3,
                             stage_index=3, start_state=args.state)
    env.reset(seed=0)
    nes = env._nes
    blob0 = nes.save()

    # --- part 1: walk up to the boss, watch the HUD --------------------
    print("=== HUD before the boss (normal play) ===", flush=True)
    nb0, br0 = hud_stats(hud_strip(np.asarray(nes.step(1), dtype=np.uint8)))
    print(f"  non-bg px={nb0}  brightness={br0:.1f}  "
          f"stamina={int(nes[R.STAMINA])}", flush=True)

    nes.load(blob0)
    snap = None
    hud_boss = None
    ck = torch.load(os.path.join(_ROOT, "runs", "ppo_14d",
                                 "ppo_27770880.pt"),
                    map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, N_ACTIONS, 128)
    net.load_state_dict(ck["model"])
    net.eval()
    # walk there with the POLICY: a scripted "hold right" runner falls into
    # the pit in front of the boss before it ever gets there
    for f in range(args.frames):
        o = env._obs()
        with torch.no_grad():
            a = int(net({"img": torch.as_tensor(o["img"][None]),
                         "ram": torch.as_tensor(o["ram"][None],
                                                dtype=torch.float32)}
                        )[0].argmax())
        o, r, done, info = env.step(a)
        frame = (o["img"][:, :, -1] * 255).astype(np.uint8)
        if has_boss(nes):
            # obs is a 80x86 grey thumbnail, too small to judge the HUD --
            # grab one full RGB frame for the HUD comparison
            frame = np.asarray(nes.step(1), dtype=np.uint8)
            snap = nes.save()
            hud_boss = hud_stats(hud_strip(frame))
            print(f"\n=== boss ON SCREEN at frame {f} ===", flush=True)
            print(f"  HUD: non-bg px={hud_boss[0]}  "
                  f"brightness={hud_boss[1]:.1f}", flush=True)
            print(f"  change vs normal: px {hud_boss[0] - nb0:+d}  "
                  f"brightness {hud_boss[1] - br0:+.1f}", flush=True)
            print(f"  slots: {slots(nes)}", flush=True)
            break
    if snap is None:
        print("never met the boss")
        return

    # --- part 2: A/B damage test --------------------------------------
    for name, do_throw in (("A stand still", False),
                           ("B jump-throw at head", True)):
        nes.load(snap)
        series = []
        for f in range(900):
            if do_throw:
                # hop, and throw at the top of the jump so the axe arcs up
                ctrl = NES_INPUT_A if f % 30 < 6 else 0
                if f % 30 == 3:
                    ctrl |= NES_INPUT_B
            else:
                ctrl = 0
            nes.controller = ctrl
            nes.step(1)
            series.append(ram(nes))
            if int(nes[R.DEATH]) != 0:
                print(f"  {name}: died at frame {f}", flush=True)
                break
        arr = np.stack(series)          # (T, 2048)
        # bytes that only ever decrease, and move at least twice
        dec = np.zeros(N_RAM, dtype=bool)
        for a in range(N_RAM):
            col = arr[:, a].astype(np.int16)
            d = np.diff(col)
            d = d[(d != 0) & (np.abs(d) < 100)]   # ignore wrap/resets
            if len(d) >= 2 and (d < 0).all():
                dec[a] = True
        idx = np.flatnonzero(dec)
        print(f"\n=== {name}: {len(series)} frames ===", flush=True)
        print(f"  monotonically DECREASING bytes: "
              f"{[hex(int(a)) for a in idx]}", flush=True)
        for a in idx[:12]:
            col = arr[:, a]
            steps = [f"{int(col[i])}->{int(col[i + 1])}"
                     for i in range(len(col) - 1)
                     if col[i] != col[i + 1]][:14]
            print(f"    ${int(a):04X}: {' '.join(steps)}", flush=True)
        np.save(os.path.join(args.out, f"{'A' if not do_throw else 'B'}.npy"),
                arr)

    # --- part 3: which bytes differ between A and B --------------------
    pa = os.path.join(args.out, "A.npy")
    pb = os.path.join(args.out, "B.npy")
    if os.path.exists(pa) and os.path.exists(pb):
        A = np.load(pa)
        B = np.load(pb)
        T = min(len(A), len(B))
        decB = np.zeros(N_RAM, dtype=bool)
        for a in range(N_RAM):
            col = B[:T, a].astype(np.int16)
            d = np.diff(col)
            d = d[(d != 0) & (np.abs(d) < 100)]
            if len(d) >= 2 and (d < 0).all():
                decB[a] = True
        decA = np.zeros(N_RAM, dtype=bool)
        for a in range(N_RAM):
            col = A[:T, a].astype(np.int16)
            d = np.diff(col)
            d = d[(d != 0) & (np.abs(d) < 100)]
            if len(d) >= 2 and (d < 0).all():
                decA[a] = True
        only = np.flatnonzero(decB & ~decA)
        print(f"\n=== bytes that fall ONLY when throwing (boss HP?) ===",
              flush=True)
        print(f"  {[hex(int(a)) for a in only]}", flush=True)
        for a in only:
            print(f"    ${int(a):04X}: "
                  f"{list(B[:T, a].astype(int))[:40]}", flush=True)


if __name__ == "__main__":
    main()
