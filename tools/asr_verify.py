"""Closed-loop check: synth a line -> take its public URL -> run ASR on it
-> compare with the text we asked for.

This proves whether qwen-audio-3.1-tts-next actually SPEAKS our script
or hallucinates unrelated content.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import load_key  # noqa: E402

KEY = load_key()
TTS = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
ASR = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}


def post(url, body, extra=None):
    h = dict(H)
    if extra:
        h.update(extra)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    return json.load(urllib.request.urlopen(req, timeout=120))


def synth_url(text, prompt, voice=None):
    b = {"model": "qwen-audio-3.1-tts-next",
         "input": {"text": text, "text_prompt": prompt}}
    if voice:
        b["parameters"] = {"voice": voice}
    r = post(TTS, b)
    return r["output"]["audio"]["url"]


def asr(url, model="paraformer-v2"):
    r = post(ASR, {"model": model, "input": {"file_urls": [url]}},
             {"X-DashScope-Async": "enable"})
    tid = r["output"]["task_id"]
    for _ in range(40):
        time.sleep(3)
        s = json.load(urllib.request.urlopen(
            urllib.request.Request(f"https://dashscope.aliyuncs.com/api/v1/tasks/{tid}",
                                   headers={"Authorization": "Bearer " + KEY}),
            timeout=60))
        st = s["output"]["task_status"]
        if st == "SUCCEEDED":
            res = s["output"].get("results") or []
            texts = []
            for item in res:
                tu = item.get("transcription_url")
                if not tu:
                    continue
                try:
                    data = json.load(urllib.request.urlopen(tu, timeout=60))
                except Exception as e:
                    texts.append(f"<download fail {e}>")
                    continue
                if isinstance(data, dict):
                    tr = data.get("transcripts") or data.get("results") or []
                    for t in tr:
                        texts.append(t.get("text", ""))
                elif isinstance(data, list):
                    for t in data:
                        if isinstance(t, dict):
                            texts.append(t.get("text", ""))
                if not texts:
                    texts.append(json.dumps(data, ensure_ascii=False)[:300])
            return " | ".join(t for t in texts if t) or \
                json.dumps(s["output"], ensure_ascii=False)[:400]
        if st in ("FAILED", "CANCELED"):
            return f"ASR {st}: {json.dumps(s['output'], ensure_ascii=False)[:300]}"
    return "ASR timeout"


CASES = [
    ("这一次，WorkBuddy 立了大功。", None),
    ("这一次，WorkBuddy 立了大功。", "anmingyuan_v3.1"),
    ("我折腾了好几个月的冒险岛，", "anmingyuan_v3.1"),
    ("这次三天，就训练通关了。", "anmingyuan_v3.1"),
]

PROMPT = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常，发音清晰。"

if __name__ == "__main__":
    for text, voice in CASES:
        print("=" * 66)
        print("ASKED :", text, " voice:", voice)
        try:
            url = synth_url(text, PROMPT, voice)
            print("HEARD :", asr(url))
        except Exception as e:
            msg = str(e)
            if hasattr(e, "read"):
                msg = e.read().decode("utf-8", "ignore")[:250]
            print("ERROR :", msg)
