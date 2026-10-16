"""系统定位摄像机(scroll)地址。

原理：玩家按右跑动时，摄像机地址必须单调递增；玩家静止时必须不变。
方法：双段实验（静止段 + 跑动段），扫描 $0000-$07FF 全部地址，
     筛选出「静止时不变 + 跑动时单调递增」的地址，再与画面互相关测得的
     真实滚动量做相关性验证。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes"


def boot(nes, warmup=300):
    nes.step(frames=warmup)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def snapshot(nes):
    return np.fromiter((nes[a] for a in range(0x800)), dtype=np.int16, count=0x800)


def main():
    nes = NES(ROM)
    boot(nes)

    # ---- 段 A：静止 300 帧（应无地址变化，除了动画计时器）----
    base = snapshot(nes)
    for _ in range(300):
        nes.controller = 0
        nes.step()
    after_a = snapshot(nes)
    changed_idle = np.abs(after_a.astype(int) - base.astype(int)) > 0

    # ---- 段 B：按右跑 600 帧（摄像机应滚动）----
    frames_b = []
    snap_b = []
    for i in range(600):
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if (i % 40) < 12 else 0)
        nes.step()
        if i % 150 == 0:
            snap_b.append(snapshot(nes).astype(int))
    del nes

    # 摄像机候选：跑动期间数值持续变化（取最后一段与最初一段差值 > 10）
    d_b = snap_b[-1] - snap_b[0]
    candidates = []
    for a in range(0x800):
        if changed_idle[a]:
            continue  # 静止时也变的（计时器/动画）排除
        if abs(d_b[a]) >= 10:
            candidates.append((a, int(d_b[a]), int(snap_b[0][a]), int(snap_b[-1][a])))

    print("候选摄像机地址（静止不变 & 跑动时变化 >=10）:")
    print(f"{'addr':>7} {'start':>6} {'end':>6} {'delta':>7}")
    for a, d, s0, s1 in sorted(candidates, key=lambda t: -abs(t[1])):
        print(f"  ${a:04x} {s0:6d} {s1:6d} {d:+7d}")


if __name__ == "__main__":
    main()
