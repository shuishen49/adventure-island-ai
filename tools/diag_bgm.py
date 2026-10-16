"""Probe whether the model can emit a music-only track (no speech)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import synth, load_key  # noqa: E402

CASES = {
    "bgm_empty": ("", "只输出背景音乐，不要任何人声。史诗感弦乐铺底，鼓点推进，纯音乐。"),
    "bgm_dot": ("……", "只输出背景音乐，没有人声。史诗感弦乐，节奏推进，纯音乐。"),
    "bgm_hum": ("嗯", "没有人声，只有背景音乐：史诗感弦乐与鼓，纯音乐演奏。"),
}

if __name__ == "__main__":
    key = load_key()
    for tag, (text, prompt) in CASES.items():
        try:
            p = synth(text, prompt, tag, key)
            size = os.path.getsize(p)
            print(f"{tag:12s} dur={(size - 44) / 192000:6.2f}s bytes={size}")
        except Exception as e:
            print(f"{tag:12s} FAILED: {str(e)[:160]}")
