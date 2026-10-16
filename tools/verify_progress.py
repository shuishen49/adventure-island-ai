"""验证世界进度公式：
   progress = 累计(-camera_delta mod wrap) + (screen_x - screen_x0)

摄像机 $0593 每帧 -= 玩家前进量。检测 wrap（值突跳 >128）并累计。
用长跑 + 画面拼图人眼验证进度正确。
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cynes import NES, NES_INPUT_A, NES_INPUT_RIGHT, NES_INPUT_START  # noqa: E402

import ram_map as R  # noqa: E402

ROM = r"D:\source_code\adventure-island-ai\roms\Adventure Island (USA).nes"
OUT = r"D:\source_code\adventure-island-ai\outputs\verify_progress.png"


def boot(nes, warmup=300):
    nes.step(frames=warmup)
    nes.controller = NES_INPUT_START
    nes.step(frames=60)
    nes.controller = 0
    nes.step(frames=120)


def read_cam(nes):
    return nes[0x0593]


def main():
    nes = NES(ROM)
    boot(nes)

    cam_prev = read_cam(nes)
    cam_travel = 0.0
    sx0 = nes[0x0584]
    tiles = []
    marks = []

    N = 1800
    for i in range(N):
        nes.controller = NES_INPUT_RIGHT | (NES_INPUT_A if (i % 40) < 12 else 0)
        buf = np.asarray(nes.step(), dtype=np.uint8)
        cam = read_cam(nes)
        d = (cam_prev - cam) % 256          # 摄像机递减 → 正向前进
        if d > 128:                          # wrap 反向跳变（不该发生）
            d -= 256
        cam_travel += d
        cam_prev = cam
        if i % 300 == 0:
            sx = nes[0x0584]
            prog = cam_travel + (sx - sx0)
            marks.append((i, prog, nes[R.LIVES]))
            tiles.append((i, prog, Image.fromarray(buf).resize((256, 240))))
    nes.controller = 0
    del nes

    print("frame | progress_px | lives")
    for i, p, lv in marks:
        print(f"f{i:5d} | {p:10.1f} | {lv}")

    cols = len(tiles)
    sheet = Image.new("RGB", (256 * cols, 240), (0, 0, 0))
    for k, (i, p, im) in enumerate(tiles):
        sheet.paste(im, (k * 256, 0))
    sheet.save(OUT)
    print(f"\nsaved -> {OUT}")
    print("预期：进度应持续增长（约 1.5px/帧 → 1800 帧 ≈ 2500px），死亡后不应清零")


if __name__ == "__main__":
    main()
