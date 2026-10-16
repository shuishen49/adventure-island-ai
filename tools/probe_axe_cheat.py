"""Two checks before we change the env:

1. Writing weapon=1 ($006D) really arms the axe: does tapping B spawn
   the id-66 projectile?  (B is edge-triggered, so hold/release.)
2. Is $0072 (death flag) safe as an instant-death signal?
"""
import sys

sys.path.insert(0, ".")
from cynes import NES, NES_INPUT_B, NES_INPUT_RIGHT, NES_INPUT_START

import ram_map as R
from env import ROM


def boot():
    nes = NES(ROM)
    nes.step(frames=300)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=60)
    return nes


def tap_b(nes, cycles=60, hold=2, gap=4):
    """Edge-triggered B presses: hold `hold` frames, release `gap`."""
    shots = 0
    max_id_seen = set()
    for _ in range(cycles):
        nes.controller = NES_INPUT_B
        for _ in range(hold):
            nes.step(frames=1)
            ids = [nes[R.OBJ.id + s] for s in range(16)]
            max_id_seen.update(i for i in ids if i)
            if 66 in ids:
                shots += 1
        nes.controller = 0
        for _ in range(gap):
            nes.step(frames=1)
            ids = [nes[R.OBJ.id + s] for s in range(16)]
            if 66 in ids:
                shots += 1
    return shots, max_id_seen


def test_axe_cheat() -> None:
    nes = boot()
    print(f"boot: weapon={nes[R.WEAPON]}")
    s0, ids0 = tap_b(nes)
    print(f"weapon=0, tap B: id-66 frames = {s0}")

    nes[R.WEAPON] = 1
    s1, ids1 = tap_b(nes)
    print(f"weapon=1, tap B: id-66 frames = {s1}")
    print(f"weapon now = {nes[R.WEAPON]}")

    # dump the weapon-related bytes to see what the game really reads
    row = [nes[0x0060 + i] for i in range(32)]
    print("$0060..$007F:", " ".join(f"{v:02X}" for v in row))


def test_death_flag_safety() -> None:
    nes = boot()
    nes[R.WEAPON] = 1
    marks = []
    rng = __import__("numpy").random.default_rng(1)
    acts = [0x00, NES_INPUT_RIGHT, NES_INPUT_RIGHT | 0x80]
    for f in range(3000):
        nes.controller = acts[int(rng.integers(0, len(acts)))]
        nes.step(frames=1)
        if nes[R.DEATH]:
            marks.append((f, nes[R.DEATH], nes[R.LIVES],
                          nes[R.STAGE_COMPLETE]))
            break
    print(f"\n$0072 first non-zero: {marks[:1]}  "
          f"(stage_complete={marks[0][3] if marks else 'n/a'})")


if __name__ == "__main__":
    test_axe_cheat()
    test_death_flag_safety()
