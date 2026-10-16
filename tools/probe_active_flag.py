"""Probe: locate the per-slot ACTIVE flag by full-RAM diff.

Protocol:
  1. play greedily; when an enemy slot is alive (and another object slot is
     active too), dump the full 2 KB RAM  -> A
  2. keep playing until that slot becomes stale (death + respawn, all its
     fields frozen for >60 frames) while OTHER slots are active  -> B
  3. candidate = addr where A != B, excluding the known object field
     arrays; then keep only addresses whose per-slot base shows
     "active slots share one value, stale slot differs" in BOTH dumps.
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from cynes import NES

import ram_map as R
from env import ACTIONS, OBS_DIM, ROM
from enjoy import obs_tensor, reset_cam
from train import ActorCritic
from visualize import boot, objects

KNOWN_FIELD_BASES = [R.OBJ.id, R.OBJ.x_fr, R.OBJ.x_lo, R.OBJ.x_hi,
                     R.OBJ.spd_x, R.OBJ.y_fr, R.OBJ.y_lo, R.OBJ.gravity,
                     0x0604, 0x0614, 0x0624, 0x0634, 0x0644, 0x0654, 0x0664]


def dump(nes):
    return [nes[a] for a in range(0x800)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="runs/ppo/ppo_00092160.pt")
    ap.add_argument("--max-frames", type=int, default=12000)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(OBS_DIM, len(ACTIONS), 128)
    net.load_state_dict(ck["model"])
    net.eval()

    nes = NES(ROM)
    boot(nes)
    reset_cam(nes)

    action = 0
    prev_lives = nes[R.LIVES]
    dump_a = None
    watch = None
    frozen = 0
    dump_b = None
    frozen_sig = None

    for i in range(args.max_frames):
        if i % 4 == 0:
            with torch.no_grad():
                action = int(net(obs_tensor(nes))[0].argmax())
        nes.controller = ACTIONS[action][0]
        nes.step()
        lv = nes[R.LIVES]
        objs = objects(nes)
        enemies = [o for o in objs if o[1] in R.ENEMY_IDS and o[0] != 0]
        n_active = sum(1 for o in objs)

        if dump_a is None:
            if enemies and n_active >= 3:
                watch = enemies[0][0]
                watch_id = nes[R.OBJ.id + watch]
                dump_a = dump(nes)
                print(f"f{i}: dump A (slot {watch} id {watch_id}, "
                      f"{n_active} active slots)")
                prev_lives = lv
            continue

        if dump_b is None:
            # signature of watched slot (id included: a reused slot would
            # get a new id and reset the frozen counter)
            sig = tuple(nes[b + R.OBJ.STRIDE * watch] for b in KNOWN_FIELD_BASES)
            if sig == frozen_sig:
                frozen += 1
            else:
                frozen = 0
                frozen_sig = sig
            if lv < prev_lives:
                print(f"f{i}: death (lives={lv})")
            prev_lives = lv
            if frozen > 90 and nes[R.DEATH] == 0:
                if nes[R.OBJ.id + watch] != watch_id:
                    print(f"f{i}: slot {watch} reused (id now "
                          f"{nes[R.OBJ.id + watch]}), still waiting")
                    frozen = 0
                    continue
                dump_b = dump(nes)
                print(f"f{i}: dump B (slot {watch} stale, frozen {frozen}f)")
                break

    nes.controller = 0
    if dump_b is None:
        print("no stale state reached, abort")
        return

    print(f"\ndiff A->B (slot {watch}, {len(KNOWN_FIELD_BASES)} known "
          f"field arrays excluded):")
    cands = []
    for addr in range(0x800):
        if any(addr - R.OBJ.STRIDE * watch == b for b in KNOWN_FIELD_BASES):
            continue
        if dump_a[addr] != dump_b[addr]:
            cands.append(addr)
    print(f"raw diffs: {len(cands)}")

    # structural filter: per-slot array where, in each dump, all ACTIVE
    # slots share one value and the stale slot differs.
    active_slots = [o[0] for o in objects(nes)]
    print(f"slots active at dump-B end: {active_slots}")
    hits = []
    for addr in cands:
        base = addr - R.OBJ.STRIDE * watch
        if base < 0 or base + 15 >= 0x800:
            continue
        va = [dump_a[base + s] for s in range(16)]
        vb = [dump_b[base + s] for s in range(16)]
        # in A the watch slot was active; in B it is stale
        # require: A[watch]==values of another active slot in A,
        #           B[watch]!=values of active slots in B
        others_a = [va[s] for s in range(16) if s != watch]
        others_b = [vb[s] for s in active_slots if s != watch]
        if va[watch] in others_a and vb[watch] not in others_b:
            hits.append((base, va[watch], vb[watch],
                         [vb[s] for s in active_slots]))

    print("\nstructural hits (base, A_val=active, B_val=stale, B active vals):")
    for base, va_w, vb_w, act in hits:
        print(f"  ${base:04X}: active={va_w} stale={vb_w} "
              f"active_slots_now={act}")
    if not hits:
        print("  (none)")
    del nes


if __name__ == "__main__":
    main()
