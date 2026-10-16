"""Verify: does splitting long narration into short sentences fix the rate?"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import synth, load_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tts_diag2")
os.makedirs(OUT, exist_ok=True)
PROMPT = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常。"

CASES = {
    "c08": "这次三天就通关了",                       # 8
    "c12": "我折腾了几个月的冒险岛",                  # 12
    "c16": "这一次，WorkBuddy 立了大功了。",          # ~16
    "c22": "最开始我想得很简单，把坐标喂给 PPO。",      # 21
    "c28": "结果 AI 连平台在哪都不知道，地形在画面上。", # 23
    "c40": "最开始我想得很简单：把内存里的玩家坐标、敌人位置、速度抓出来喂给 PPO，不就行了？",
}

if __name__ == "__main__":
    key = load_key()
    for tag, text in CASES.items():
        p = synth(text, PROMPT, tag, key)
        size = os.path.getsize(p)
        dur = (size - 44) / 192000
        chars = len(text)
        print(f"{tag} chars={chars:3d} dur={dur:5.2f}s rate={chars / dur:5.2f} 字/秒")
        time.sleep(0.4)
