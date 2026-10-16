"""voice_preview_v5.py — compare candidate voices with the v5 emotional style.

Synthesises the hook lines with every candidate voice using the exact style
build_v5.py uses, so the comparison matches the final narration.
Outputs: outputs/preview/v5/voice_<id>.mp3 + v5_voices_compare.mp3
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import END, MODEL, load_key  # noqa: E402
import imageio_ffmpeg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "preview", "v5")
SR = 48000

STYLE = (
    "成熟的中文男声纪录片旁白，声音磁性低沉有质感，情绪饱满富有感染力，"
    "语气笃定从容、张弛有度，像讲述一段亲身经历的技术故事，节奏自然，发音清晰。"
)
TEXT = "这一次，WorkBuddy 立了大功。我折腾了好几个月的冒险岛，这次三天，就训练通关了。"

VOICES = [
    ("longsanshu_v3.1", "龙三叔"),
    ("longanzhi_v3.1", "龙安志"),
    ("xunanchuan_v3.1", "许南川"),
    ("wenhuaizhi_v3.1", "闻怀之"),
    ("huozhuoshi_v3.1", "霍拙石"),
    ("anmingyuan_v3.1", "安明远"),
]


def synth(voice, dst, key):
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": TEXT},
        "parameters": {"voice": voice, "instruction": STYLE},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        END, data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    urllib.request.urlretrieve(r["output"]["audio"]["url"], dst)


def to_mp3(wav):
    mp3 = wav[:-4] + ".mp3"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", wav,
                    "-c:a", "libmp3lame", "-b:a", "192k", mp3],
                   capture_output=True)
    return mp3


def silence(path, sec=0.9):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.zeros((int(sec * SR), 2), np.int16).tobytes())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    key = load_key()
    wavs = []
    for vid, cn in VOICES:
        wav = os.path.join(OUT, f"voice_{vid}.wav")
        try:
            synth(vid, wav, key)
            to_mp3(wav)
            wavs.append(wav)
            print(f"{vid:20s} {cn}  ok  {os.path.getsize(wav) // 1024} KB")
        except Exception as e:
            msg = str(e)
            if hasattr(e, "read"):
                msg = e.read().decode("utf-8", "ignore")[:160]
            print(f"{vid:20s} {cn}  FAILED {msg}")
        time.sleep(0.3)

    if wavs:
        sil = os.path.join(OUT, "sil.wav")
        silence(sil)
        lst = os.path.join(OUT, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for w in wavs:
                f.write("file '%s'\n" % sil)
                f.write("file '%s'\n" % w)
        cmp_mp3 = os.path.join(OUT, "v5_voices_compare.mp3")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c:a", "libmp3lame",
                        "-b:a", "192k", cmp_mp3], capture_output=True)
        os.remove(lst)
        print("compare:", cmp_mp3)
