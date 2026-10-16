"""直接看画面：按住右跑 1200 帧，每 150 帧截一张，拼图输出。

目的：判断玩家到底是在前进（背景滚动）还是真的被挡住/掉坑反复重生。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "orig_run_sheet.png")


def boot(nes, warmup=300):
    nes.step(frames=warmup)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def main():
    nes = NES(ROM)
    boot(nes)
    tiles, info = [], []
    for i in range(1200):
        # 全程按右，并周期性跳跃（尽力不掉坑）
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if (i % 40) < 12 else 0)
        buf = np.asarray(nes.step(), dtype=np.uint8)
        if i % 150 == 0:
            x = R.px(R.read_obj_world_x(nes, 0))
            y = R.px(R.read_obj_world_y(nes, 0))
            info.append((i, x, y, nes[R.LIVES]))
            img = Image.fromarray(buf).resize((256 * 2, 240 * 2), Image.NEAREST)
            tiles.append(img)
    del nes

    cols, rows = 4, 2
    sheet = Image.new("RGB", (256 * 2 * cols, 240 * 2 * rows), (20, 20, 20))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * 512, (k // cols) * 480))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sheet.save(OUT)

    print("frame    x        y      lives")
    for i, x, y, lv in info:
        print(f"  f{i:4d}  {x:7.1f}  {y:6.1f}   {lv}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
