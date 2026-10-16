"""Try parameter placements to get normal, length-proportional speech."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import load_key  # noqa: E402

END = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
MODEL = "qwen-audio-3.1-tts-next"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tts_diag3")
os.makedirs(OUT, exist_ok=True)

LONG = "这一次，WorkBuddy 立了大功。我折腾了好几个月的《冒险岛》，这次三天，就训练通关了。"
SHORT = "这一次，WorkBuddy 立了大功。"
PROMPT = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常。"


def call(tag, text, prompt=PROMPT, input_extra=None, parameters=None):
    body = {"model": MODEL, "input": {"text": text, "text_prompt": prompt}}
    if input_extra:
        body["input"].update(input_extra)
    if parameters:
        body["parameters"] = parameters
    try:
        req = urllib.request.Request(
            END, data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + load_key(),
                     "Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=120))
        a = r["output"]["audio"]
        p = os.path.join(OUT, tag + ".wav")
        urllib.request.urlretrieve(a["url"], p)
        chars = len(text)
        d = a.get("duration", 0)
        print(f"{tag:22s} chars={chars:3d} dur={d:6.2f}s rate={chars / d:5.2f} 字/秒")
    except Exception as e:
        msg = str(e)
        if hasattr(e, "read"):
            try:
                msg = e.read().decode("utf-8", "ignore")[:220]
            except Exception:
                pass
        print(f"{tag:22s} FAILED: {msg}")


if __name__ == "__main__":
    print("=== baseline (no voice) ===")
    call("base_long", LONG)
    call("base_short", SHORT)

    print("\n=== voice in input ===")
    for v in ["anmingyuan_v3.1", "huozhuoshi_v3.1", "longsanshu_v3.1"]:
        call("in_voice_" + v.split("_")[0], LONG, input_extra={"voice": v})

    print("\n=== voice in parameters ===")
    call("param_voice", LONG, parameters={"voice": "anmingyuan_v3.1"})

    print("\n=== rate control ===")
    for r in [0.8, 1.0]:
        call(f"param_rate_{r}", LONG,
             input_extra={"voice": "anmingyuan_v3.1"},
             parameters={"rate": r})

    print("\n=== format wav / sample rate ===")
    call("param_fmt", LONG, input_extra={"voice": "anmingyuan_v3.1"},
         parameters={"format": "wav", "sample_rate": 24000, "rate": 1.0})
