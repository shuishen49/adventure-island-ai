"""Interactive RAM probe: locate player / enemy / physics addresses.

Strategy
--------
1. Boot the ROM and press START until the title screen is left.
2. Run a controlled experiment:
      phase A - stand still
      phase B - hold RIGHT
      phase C - hold RIGHT + jump repeatedly
   An address that changes a lot in B but is quiet in A is a candidate
   for the horizontal position / scroll. Vertical motion isolates Y.
3. Print a ranked candidate table per phase.
"""

from collections import Counter

import numpy as np

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT

ROM = r"D:\source_code\cynes\冒险岛1.nes"
RAM_END = 0x800


def scan(nes):
    return np.fromiter((nes[a] for a in range(RAM_END)), dtype=np.uint8, count=RAM_END)


def boot(nes, frames_after_start=180):
    """Press START until we leave the attract/title screen, then settle."""
    for _ in range(8):
        nes.controller = 0x10  # START
        nes.step()
    nes.controller = 0
    for _ in range(frames_after_start):
        nes.step()


def observe(nes, controller, frames):
    """Run with a fixed controller and return the per-address activity stats."""
    nes.controller = controller
    snaps = []
    for _ in range(frames):
        nes.step()
        snaps.append(scan(nes))
    nes.controller = 0
    return np.stack(snaps)


def stats(snaps):
    """Per-address: #unique values, #changes, monotonic-ish score."""
    out = {}
    for addr in range(RAM_END):
        col = snaps[:, addr]
        uniq = len(np.unique(col))
        changes = int((np.diff(col.astype(np.int16)) != 0).sum())
        if changes == 0:
            continue
        d = np.diff(col.astype(np.int16))
        mono = abs(d.sum()) / (np.abs(d).sum() + 1e-9)
        out[addr] = dict(uniq=uniq, changes=changes, mono=float(mono),
                         lo=int(col.min()), hi=int(col.max()),
                         last=int(col[-1]))
    return out


def report(name, st, top=28):
    print(f"\n{'=' * 78}\n{name}  ({len(st)} active addresses)\n{'=' * 78}")
    print(f"{'addr':>6} {'uniq':>5} {'chg':>5} {'mono':>5} {'lo':>4} {'hi':>4} {'last':>5}")
    for addr in sorted(st, key=lambda a: -st[a]["changes"])[:top]:
        s = st[addr]
        print(f"{addr:#06x} {s['uniq']:5d} {s['changes']:5d} {s['mono']:5.2f} "
              f"{s['lo']:4d} {s['hi']:4d} {s['last']:5d}")


def main():
    nes = NES(ROM)
    boot(nes)

    # sanity: did we actually get into gameplay?
    print("post-boot snapshot of known Action-Replay addresses:")
    for a, label in [(0x3F, "lives"), (0x76, "time/health"), (0x6D, "weapon"),
                     (0x6E, "skateboard"), (0x75, "invincible"),
                     (0x37, "area"), (0x38, "round"), (0x39, "checkpoint")]:
        print(f"  {a:#06x} {label:<12} = {nes[a]:3d}")

    a = observe(nes, 0x00, 240)
    b = observe(nes, NES_INPUT_RIGHT, 240)
    c = observe(nes, NES_INPUT_RIGHT | NES_INPUT_A, 240)

    sa, sb, sc = stats(a), stats(b), stats(c)
    report("A: standing still", sa)
    report("B: holding RIGHT", sb)
    report("C: RIGHT + jump spam", sc)

    # addresses that light up only when moving right -> horizontal position
    print(f"\n{'=' * 78}\nB-only movers (quiet when idle, busy when running right)\n{'=' * 78}")
    print(f"{'addr':>6} {'A_chg':>6} {'B_chg':>6} {'mono':>5} {'lo':>4} {'hi':>4} {'last':>5}")
    cands = [x for x in sb if sa.get(x, {}).get("changes", 0) < 5]
    for addr in sorted(cands, key=lambda x: -sb[x]["changes"])[:30]:
        s = sb[addr]
        print(f"{addr:#06x} {sa.get(addr, {}).get('changes', 0):6d} {s['changes']:6d} "
              f"{s['mono']:5.2f} {s['lo']:4d} {s['hi']:4d} {s['last']:5d}")

    del nes


if __name__ == "__main__":
    main()
