"""Verify the fix: split narration into short sentences, synthesise each,
then concatenate. Checks that per-sentence rate is sane and total is sane.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import synth, load_key  # noqa: E402

PROMPT = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常，发音清晰。"

GROUPS = {
    "g1": [
        "这一次，WorkBuddy 立了大功。",
        "我折腾了好几个月的冒险岛，",
        "这次三天，就训练通关了。",
    ],
    "g2": [
        "最开始我想得很简单，",
        "把内存里的坐标和速度抓出来，",
        "喂给 PPO 就行了？",
        "结果 AI 连平台在哪都不知道。",
    ],
}

if __name__ == "__main__":
    key = load_key()
    for g, lines in GROUPS.items():
        tot = 0.0
        print(f"--- {g} ---")
        for i, line in enumerate(lines):
            tag = f"{g}_{i}"
            p = synth(line, PROMPT, tag, key)
            size = os.path.getsize(p)
            dur = (size - 44) / 192000
            chars = len(line)
            tot += dur
            print(f"  {tag} {chars:3d}字 {dur:5.2f}s  {chars / dur:5.2f} 字/秒  | {line}")
            time.sleep(0.3)
        chars = sum(len(x) for x in lines)
        print(f"  TOTAL {chars}字 {tot:.2f}s  → 平均 {chars / tot:.2f} 字/秒")
