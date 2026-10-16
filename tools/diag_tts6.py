"""Measure speech rate (after trimming the music intro) for short sentences
synthesised WITH a background-music instruction.
"""
import os
import sys
import time

import numpy as np
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import synth, load_key  # noqa: E402

PROMPT = (
    "成熟的中文男声纪录片旁白，语气自然笃定，语速正常，发音清晰。"
    "背景音乐：史诗感弦乐铺底，音量很低，不盖过人声。"
)

LINES = [
    "这一次，WorkBuddy 立了大功。",
    "我折腾了好几个月的冒险岛。",
    "这次三天就训练通关了。",
    "结果 AI 连平台在哪都不知道。",
    "改成多进程之后快了十二倍。",
]


def speech_span(path):
    w = wave.open(path)
    n = w.getnframes()
    d = np.frombuffer(w.readframes(n), dtype=np.int16)
    d = d.reshape(-1, w.getnchannels()).astype(np.float32) / 32768
    mono = d.mean(1)
    seg = 2400  # 0.05 s
    vals = [20 * np.log10(max(np.sqrt((mono[i:i + seg] ** 2).mean()), 1e-9))
            for i in range(0, len(mono) - seg, seg)]
    loud = [i for i, v in enumerate(vals) if v > -25]
    if not loud:
        return 0.0, len(mono) / 48000
    return loud[0] * 0.05, (loud[-1] + 1) * 0.05


if __name__ == "__main__":
    key = load_key()
    for i, line in enumerate(LINES):
        tag = f"rate_{i}"
        p = synth(line, PROMPT, tag, key)
        total = (os.path.getsize(p) - 44) / 192000
        s, e = speech_span(p)
        span = e - s
        chars = len(line)
        print(f"{tag} {chars:3d}字  total={total:5.2f}s  人声段={s:4.2f}-{e:4.2f} "
              f"({span:4.2f}s)  裁剪后语速={chars / span:5.2f} 字/秒")
        time.sleep(0.3)
