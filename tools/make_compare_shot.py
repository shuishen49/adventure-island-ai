"""Side-by-side screenshot: stage 1-1 start vs the two round 1-2 states.

Exists so the stages can be eyeballed instead of trusted: the second
column is the RAM warp, the third is what the game really reaches after
the trained policy clears 1-1.  They should look like the same screen.

usage: python tools/make_compare_shot.py
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
SRC = os.path.join(_ROOT, "outputs", "probe_round2")
OUT = os.path.join(SRC, "compare_stages.png")

PANELS = [
    ("S11_raw.png", "第一关 1-1 开局\n(round=0, 出生 80,152)"),
    ("warp_round2_start.png", "第二关 1-2 开局 · RAM 扭曲\n(round=1, 出生 80,80)"),
    ("natural_round2_start.png", "第二关 1-2 开局 · 模型真通关\n(round=1, 出生 80,80)"),
]

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]


def font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def main():
    imgs = []
    for name, _ in PANELS:
        im = Image.open(os.path.join(SRC, name)).convert("RGB")
        imgs.append(im.resize((512, 480), Image.NEAREST))

    w, h = imgs[0].size
    gap = 16
    head = 46
    total = Image.new("RGB", (len(imgs) * w + (len(imgs) + 1) * gap,
                              h + head + 70), (245, 245, 245))
    d = ImageDraw.Draw(total)
    f_title = font(22)
    f_small = font(15)

    for i, (im, (_, caption)) in enumerate(zip(imgs, PANELS)):
        x = gap + i * (w + gap)
        total.paste(im, (x, head))
        d.rectangle([x - 1, head - 1, x + w, head + h], outline=(60, 60, 60))
        for j, line in enumerate(caption.split("\n")):
            d.text((x + 4, head + h + 8 + j * 20), line,
                   fill=(30, 30, 30), font=f_small)

    d.text((gap, 10), "冒险岛 关卡起始画面对比（左右两列应完全一致）",
           fill=(20, 20, 20), font=f_title)
    total.save(OUT)
    print("wrote", OUT, total.size)


if __name__ == "__main__":
    main()
