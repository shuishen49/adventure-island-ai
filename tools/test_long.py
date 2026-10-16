"""Re-test long-text synthesis with the CORRECT parameter shape.

Earlier "model can only read ~15 chars" was measured with the WRONG shape
(text ignored, model improvising). Re-measure properly:
  - per-scene text (44-72 chars) and the WHOLE script (529 chars)
  - magnetic male voices
  - instruction that asks for background music
Every result is verified with ASR.
"""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_verify import asr  # noqa: E402
from tts_batch import END, MODEL, load_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")
OUT = os.path.join(ROOT, "outputs", "preview")
os.makedirs(OUT, exist_ok=True)

BASE_STYLE = "成熟的中文男声纪录片旁白，声音磁性低沉有质感，语气笃定从容，发音清晰。"
BGM = "背景音乐：史诗感的管弦乐，弦乐铺底加鼓点推进，音量压低不盖过人声。"


def synth(text, style, voice, name):
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": text},
        "parameters": {"voice": voice, "instruction": style},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        END, data=body,
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))
    a = r["output"]["audio"]
    dst = os.path.join(OUT, name + ".wav")
    urllib.request.urlretrieve(a["url"], dst)
    return dst, a.get("duration"), a["url"]


def dur_of(path):
    try:
        w = wave.open(path)
        n = w.getnframes()
        return n / w.getframerate()
    except Exception:
        return -1


def to_mp3(wav):
    mp3 = wav[:-4] + ".mp3"
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", wav,
                    "-c:a", "libmp3lame", "-b:a", "192k", mp3],
                   capture_output=True)
    return mp3


if __name__ == "__main__":
    KEY = load_key()
    script = json.load(open(SCRIPT, encoding="utf-8"))
    scene_text = {s["id"]: "".join(s["lines"]) for s in script}
    full = "".join(scene_text[s["id"]] for s in script)

    jobs = []
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"

    if mode == "probe":
        # length ladder, magnetic voice, WITH bgm instruction
        jobs = [
            ("L1_s01_44", scene_text["s01"], "longsanshu_v3.1", BASE_STYLE + BGM),
            ("L2_s05_70", scene_text["s05"], "longsanshu_v3.1", BASE_STYLE + BGM),
            ("L3_all_529", full, "longsanshu_v3.1", BASE_STYLE + BGM),
        ]
    elif mode == "voice":
        # magnetic voice comparison on the same 44-char text
        jobs = [
            ("V_longsanshu", scene_text["s01"], "longsanshu_v3.1", BASE_STYLE),
            ("V_longanzhi", scene_text["s01"], "longanzhi_v3.1", BASE_STYLE),
            ("V_huozhuoshi", scene_text["s01"], "huozhuoshi_v3.1", BASE_STYLE),
            ("V_anmingyuan", scene_text["s01"], "anmingyuan_v3.1", BASE_STYLE),
        ]
    elif mode == "bgm":
        jobs = [
            ("B_with", scene_text["s01"], "longsanshu_v3.1", BASE_STYLE + BGM),
            ("B_without", scene_text["s01"], "longsanshu_v3.1", BASE_STYLE),
        ]

    for name, text, voice, style in jobs:
        print("=" * 70)
        print(f"{name}  {len(text)}字  voice={voice}  bgm={'Y' if BGM[:4] in style else 'N'}")
        try:
            path, dur, url = synth(text, style, voice, name)
        except Exception as e:
            msg = str(e)
            if hasattr(e, "read"):
                msg = e.read().decode("utf-8", "ignore")[:300]
            print("  SYNTH FAIL:", msg)
            continue
        real = dur_of(path)
        print(f"  audio: api_dur={dur}  wav_dur={real:.2f}s  "
              f"{len(text)/max(real,0.01):.2f} 字/秒")
        mp3 = to_mp3(path)
        print("  mp3:", mp3, os.path.getsize(mp3) // 1024, "KB")
        heard = asr(url)
        print(f"  HEARD ({len(heard)}字): {heard[:260]}")
