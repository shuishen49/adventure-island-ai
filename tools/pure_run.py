"""无死亡跑动实验：纯按右 500 帧（开局是平地不会死），每 100 帧存图。

人眼对比背景滚动量 + 打印候选地址轨迹。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes"
OUT = r"D:\source_code\adventure-island-ai\outputs\orig_pure_run.png"

WATCH = [0x0574, 0x0584, 0x0593, 0x0594, 0x003C, 0x00EC, 0x00ED, 0x00EE, 0x05B3]


def boot(nes, warmup=300):
    nes.step(frames=warmup)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def main():
    nes = NES(ROM)
    boot(nes)
    tiles, rows = [], []
    for i in range(500):
        nes.controller = NES_INPUT_RIGHT
        buf = np.asarray(nes.step(), dtype=np.uint8)
        if i % 100 == 0:
            row = [nes[a] for a in WATCH]
            rows.append((i,) + tuple(row))
            tiles.append(Image.fromarray(buf).resize((512, 480), Image.NEAREST))
    nes.controller = 0
    del nes

    sheet = Image.new("RGB", (512 * len(tiles), 480), (0, 0, 0))
    for k, im in enumerate(tiles):
        sheet.paste(im, (k * 512, 0))
    sheet.save(OUT)

    hdr = "frame  " + "  ".join(f"${a:04x}" for a in WATCH)
    print(hdr)
    for r in rows:
        print(f"f{r[0]:4d}  " + "  ".join(f"{v:5d}" for v in r[1:]))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
