"""逐帧实验：热身 100 帧（摄像机开始滚动后），逐帧记录
$0593/$003C/$0574/$0584 与画面，用短程互相关测每帧真实滚动。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

ROM = r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes"

WATCH = [0x0574, 0x0584, 0x0593, 0x0594, 0x003C, 0x00EC, 0x00ED, 0x00EE]


def boot(nes, warmup=300):
    nes.step(frames=warmup)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def gray(f):
    return f.astype(np.float64).mean(axis=2)


def shift_of(a, b, rng=range(0, 9)):
    """找 s: b ≈ a 左移 s（摄像机右移 s）。"""
    best, bs = 1e18, 0
    for s in rng:
        d = a[:, s:] - b[:, :256 - s]
        v = float((d * d).mean())
        if v < best:
            best, bs = v, s
    return bs, best


def main():
    nes = NES(ROM)
    boot(nes)
    # 热身：跑到摄像机滚动状态
    nes.controller = NES_INPUT_RIGHT
    nes.step(frames=120)

    prev = None
    print("frame | scroll_px | " + " | ".join(f"${a:04x}" for a in WATCH))
    log = []
    for i in range(120):
        buf = np.asarray(nes.step(), dtype=np.uint8)
        vals = [nes[a] for a in WATCH]
        g = gray(buf.astype(np.float64))
        if prev is not None:
            s, res = shift_of(prev, g)
        else:
            s, res = 0, 0
        prev = g
        log.append((i, s, res, vals))
    nes.controller = 0
    del nes

    for i, s, res, vals in log[::10]:
        print(f"f{i:4d} | {s:+d} (res {res:5.0f}) | " +
              " | ".join(f"{v:5d}" for v in vals))

    # 统计：每帧滚动量分布 & 各地址每帧 delta 分布
    print("\n每帧滚动量分布:", np.bincount([s for _, s, _, _ in log[1:]],
                                          minlength=9)[1:].tolist())
    for j, a in enumerate(WATCH):
        ds = [(log[k][3][j] - log[k - 1][3][j]) % 256 for k in range(1, len(log))]
        counts = np.bincount(ds, minlength=256)
        top = np.argsort(counts)[::-1][:3]
        print(f"${a:04x} 每帧 delta 分布 top3: " +
              ", ".join(f"{d}:{counts[d]}" for d in top))


if __name__ == "__main__":
    main()
