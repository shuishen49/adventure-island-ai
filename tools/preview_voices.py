"""Preview different voices on the same short line so the user can pick."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import load_key  # noqa: E402

END = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
MODEL = "qwen-audio-3.1-tts-next"
TEXT = "这一次，WorkBuddy 立了大功。"
PROMPT = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常，发音清晰。"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "preview")
os.makedirs(OUT, exist_ok=True)

VOICES = [
    ("anmingyuan_v3.1", "安明远"),
    ("huozhuoshi_v3.1", "霍拙石"),
    ("longsanshu_v3.1", "龙三叔"),
    ("xunanchuan_v3.1", "许南川"),
    ("wenhuaizhi_v3.1", "闻怀之"),
]


def call(body, timeout=90):
    req = urllib.request.Request(
        END, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


if __name__ == "__main__":
    KEY = load_key()
    base = {"model": MODEL, "input": {"text": TEXT, "text_prompt": PROMPT}}

    # 1) probe where `voice` goes
    for where in ("parameters", "input", "top"):
        b = json.loads(json.dumps(base))
        if where == "parameters":
            b["parameters"] = {"voice": "anmingyuan_v3.1"}
        elif where == "input":
            b["input"]["voice"] = "anmingyuan_v3.1"
        else:
            b["voice"] = "anmingyuan_v3.1"
        try:
            r = call(b)
            print(f"probe {where:11s} OK  dur={r['output']['audio'].get('duration')}")
        except Exception as e:
            msg = str(e)
            if hasattr(e, "read"):
                msg = e.read().decode("utf-8", "ignore")[:180]
            print(f"probe {where:11s} FAIL {msg}")
    print("-" * 50)

    for vid, cn in VOICES:
        b = json.loads(json.dumps(base))
        b["parameters"] = {"voice": vid}
        try:
            r = call(b)
            a = r["output"]["audio"]
            dst = os.path.join(OUT, f"voice_{vid}.wav")
            urllib.request.urlretrieve(a["url"], dst)
            size = os.path.getsize(dst)
            dur = (size - 44) / 192000
            print(f"{vid:20s} {cn}  dur={dur:5.2f}s  {size/1024:6.0f}KB")
        except Exception as e:
            msg = str(e)
            if hasattr(e, "read"):
                msg = e.read().decode("utf-8", "ignore")[:180]
            print(f"{vid:20s} {cn}  FAILED {msg}")
