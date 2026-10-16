"""Batch TTS with qwen-audio-3.1-tts-next (DashScope SpeechSynthesizer API).

CRITICAL (learned the hard way, verified with ASR round-trip):
  `input.text_prompt` is the text that gets **spoken**.
  `text` is IGNORED for synthesis (it only satisfies a non-empty check).
  If you put a *style description* into text_prompt, the model will
  IMPROVISE its own sentences instead of reading your script -- it produced
  "岁月沉淀下的文明脉络..." when asked to read about WorkBuddy.

  Correct shape:
    input.text_prompt      = the script line to speak
    parameters.instruction = style / tone / pacing  (optional)
    parameters.voice       = e.g. "anmingyuan_v3.1"
"""
import json
import os
import sys
import time
import urllib.request

END = "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
MODEL = "qwen-audio-3.1-tts-next"
DEFAULT_VOICE = "anmingyuan_v3.1"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs", "tts")
os.makedirs(OUT, exist_ok=True)


def load_key():
    cfg = os.path.join(os.path.expanduser("~"), ".bailian", "config.json")
    with open(cfg) as f:
        return json.load(f)["api_key"]


def synth(text, prompt, name, key, retries=3, voice=DEFAULT_VOICE):
    path = os.path.join(OUT, name + ".wav")
    if os.path.exists(path) and os.path.getsize(path) > 2000:
        print("  skip (exists):", name)
        return path
    body = json.dumps({
        "model": MODEL,
        # text_prompt is what is READ -- never put the style description here
        "input": {"text_prompt": text},
        "parameters": {"voice": voice, "instruction": prompt},
    }, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                END, data=body,
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            url = r["output"]["audio"]["url"]
            dur = r["output"]["audio"].get("duration", 0)
            urllib.request.urlretrieve(url, path)
            print(f"  ok {name}: {dur:.2f}s  {os.path.getsize(path)} bytes")
            return path
        except Exception as e:
            print(f"  retry {attempt + 1} {name}: {e}")
            time.sleep(3)
    raise RuntimeError("failed: " + name)


def main():
    lines = json.load(open(sys.argv[1], encoding="utf-8"))
    key = load_key()
    for item in lines:
        print("->", item["name"])
        synth(item["text"], item["prompt"], item["name"], key)


if __name__ == "__main__":
    main()
