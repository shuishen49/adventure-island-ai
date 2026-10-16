"""Find the parameter shape that makes qwen-audio-3.1-tts-next SPEAK our
text verbatim (instead of improvising content from the style prompt).

Every candidate is verified with ASR: synth -> public URL -> ASR -> text.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import load_key  # noqa: E402
from asr_verify import asr, synth_url  # noqa: E402

KEY = load_key()
TTS = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
H = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

TEXT = "这一次，WorkBuddy 立了大功。"
STYLE = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常，发音清晰。"
VOICE = "anmingyuan_v3.1"


def post(body):
    req = urllib.request.Request(TTS, data=json.dumps(body).encode(), headers=H)
    return json.load(urllib.request.urlopen(req, timeout=120))


def run(tag, body):
    try:
        r = post(body)
        url = r["output"]["audio"]["url"]
    except Exception as e:
        msg = str(e)
        if hasattr(e, "read"):
            msg = e.read().decode("utf-8", "ignore")[:200]
        print(f"{tag:34s} SYNTH FAIL: {msg}")
        return
    heard = asr(url)
    ok = "WorkBuddy" in heard.replace(" ", "") or "立了大功" in heard
    print(f"{tag:34s} {'OK  ' if ok else 'MISS'} -> {heard[:70]}")


BASE = {"model": "qwen-audio-3.1-tts-next"}

if __name__ == "__main__":
    # A: text_prompt only, carrying the SCRIPT (no separate style)
    b = json.loads(json.dumps(BASE))
    b["input"] = {"text_prompt": TEXT}
    run("A input.text_prompt=TEXT", b)

    # B: text only, no prompt at all
    b = json.loads(json.dumps(BASE))
    b["input"] = {"text": TEXT}
    run("B input.text=TEXT (no prompt)", b)

    # C: text=TEXT, style in parameters.text_prompt
    b = json.loads(json.dumps(BASE))
    b["input"] = {"text": TEXT}
    b["parameters"] = {"text_prompt": STYLE, "voice": VOICE}
    run("C text + parameters.text_prompt", b)

    # D: text=TEXT, style in parameters.instruction
    b = json.loads(json.dumps(BASE))
    b["input"] = {"text": TEXT}
    b["parameters"] = {"instruction": STYLE, "voice": VOICE}
    run("D text + parameters.instruction", b)

    # E: text=TEXT, style in input.instruction
    b = json.loads(json.dumps(BASE))
    b["input"] = {"text": TEXT, "instruction": STYLE}
    b["parameters"] = {"voice": VOICE}
    run("E text + input.instruction", b)

    # F: text_prompt=TEXT (script) + parameters.instruction (style)
    b = json.loads(json.dumps(BASE))
    b["input"] = {"text_prompt": TEXT}
    b["parameters"] = {"instruction": STYLE, "voice": VOICE}
    run("F text_prompt=TEXT + instr", b)

    # G: text=TEXT + voice only (no style instruction at all)
    b = json.loads(json.dumps(BASE))
    b["input"] = {"text": TEXT}
    b["parameters"] = {"voice": VOICE}
    run("G text + voice (no style)", b)
