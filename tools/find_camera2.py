"""决定性实验：测画面真实滚动量，找出与之同步的 RAM 地址。

方法：每 60 帧同时保存 (全部RAM快照, 画面灰度图)。
相邻画面做水平互相关得到真实滚动像素。
然后找 RAM 地址序列与累积滚动最匹配的（ pearson 相关 + 数值比例）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

ROM = r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes"

PERIOD = 60
NSEG = 10  # 600 帧


def boot(nes, warmup=300):
    nes.step(frames=warmup)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def gray(frame):
    return frame.mean(axis=2)


def horiz_shift(a, b, max_shift=48):
    """返回 s 使得 b ≈ a 左移 s（摄像机右移 s px）。"""
    best, best_s = 1e18, 0
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            d = a[:, s:] - b[:, :256 - s]
        else:
            d = a[:, :256 + s] - b[:, -s:]
        v = float((d * d).mean())
        if v < best:
            best, best_s = v, s
    return best_s, best


def main():
    nes = NES(ROM)
    boot(nes)

    snaps, frames = [], []
    for i in range(NSEG * PERIOD):
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if (i % 40) < 12 else 0)
        buf = np.asarray(nes.step(), dtype=np.float64)
        if i % PERIOD == 0:
            snaps.append(np.fromiter((nes[a] for a in range(0x800)),
                                     dtype=np.int32, count=0x800))
            frames.append(buf)
    del nes

    # 相邻画面平移
    shifts = []
    for k in range(1, len(frames)):
        s, res = horiz_shift(gray(frames[k - 1]), gray(frames[k]))
        shifts.append(s)
        print(f"seg {k}: 摄像机右移 {s:+d} px (残差 {res:.0f})")
    cum = np.concatenate([[0], np.cumsum(shifts)]).astype(float)
    print("累积滚动:", cum.astype(int).tolist())

    # RAM 地址与累积滚动相关
    arr = np.array(snaps)  # (NSEG, 0x800)
    print("\n与累积滚动高度相关的地址（corr>0.99，且非全零）:")
    hits = []
    for a in range(0x800):
        col = arr[:, a].astype(float)
        if col.std() < 1e-9:
            continue
        cc = np.corrcoef(col, cum)[0, 1]
        if cc > 0.99:
            # 拟合比例: cum ≈ k * (col - col[0]) + wrap 处理不可靠，直接报 delta
            hits.append((a, cc, col[0], col[-1], col[-1] - col[0]))
    hits.sort(key=lambda t: -t[1])
    for a, cc, v0, v1, d in hits:
        print(f"  ${a:04x}  corr={cc:.4f}  {v0:5.0f} -> {v1:5.0f}  (d={d:+.0f})")


if __name__ == "__main__":
    main()
