"""Diagnose: does the BGM instruction slow down / fragment the speech?

Generates three variants of the same line and reports duration plus
the energy profile (how much of the track is actual speech).
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import synth, load_key  # noqa: E402

END = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
MODEL = "qwen-audio-3.1-tts-next"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tts_diag")
os.makedirs(OUT, exist_ok=True)

TEXT = "这一次，WorkBuddy 立了大功。我折腾了好几个月的《冒险岛》，这次三天，就训练通关了。"

VARIANTS = {
    # A: pure narration, no music instruction
    "A_plain": "成熟的中文男声解说，语速正常，语气自然笃定。",
    # B: current production prompt (with BGM)
    "B_bgm": (
        "成熟的中文男声解说，发音标准清晰，语速中等偏快，声音有磁性，"
        "叙事感强，像在讲一段自己亲身经历的技术故事。"
        "语气笃定而克制，第一句要有分量，结尾带一点压抑后的释放感。"
        "背景音乐：史诗感弦乐缓缓推进，鼓点进入，音量低不盖人声。"
    ),
    # C: named voice + no BGM
    "C_voice": "成熟的中文男声，自然流畅，语速正常。",
}


def call(text, prompt, voice=None, tag=""):
    body = {"model": MODEL, "input": {"text": text, "text_prompt": prompt}}
    if voice:
        body["input"]["voice"] = voice
    req = urllib.request.Request(
        END, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + load_key(),
                 "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    url = r["output"]["audio"]["url"]
    dur = r["output"]["audio"].get("duration", 0)
    path = os.path.join(OUT, tag + ".wav")
    urllib.request.urlretrieve(url, path)
    print(f"{tag:12s} reported_dur={dur:6.2f}s  bytes={os.path.getsize(path)}")
    return path, dur


if __name__ == "__main__":
    for tag, prompt in VARIANTS.items():
        try:
            call(TEXT, prompt, tag=tag)
        except Exception as e:
            print(f"{tag:12s} FAILED: {e}")
