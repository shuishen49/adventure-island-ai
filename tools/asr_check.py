"""Transcribe a generated TTS clip back to text, to verify the model
actually speaks the script (vs. hallucinating unrelated content)."""
import json
import mimetypes
import os
import sys
import uuid
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import load_key  # noqa: E402

KEY = load_key()
BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def transcribe(path, model="qwen-audio-asr"):
    boundary = "----wb" + uuid.uuid4().hex
    with open(path, "rb") as f:
        audio = f.read()
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "audio/mpeg"

    def field(n, v):
        return (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n').encode()

    def file_field(n, fn, data, ct):
        return (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{n}"; filename="{fn}"\r\n'
                f"Content-Type: {ct}\r\n\r\n").encode() + data + b"\r\n"

    body = (field("model", model)
            + file_field("file", name, audio, ctype)
            + f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{BASE}/audio/transcriptions", data=body,
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
        return r.get("text") or json.dumps(r, ensure_ascii=False)[:400]
    except Exception as e:
        msg = str(e)
        if hasattr(e, "read"):
            msg = e.read().decode("utf-8", "ignore")[:300]
        return f"FAIL[{model}] {msg}"


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "outputs/preview/voice_0to10s.mp3",
        "outputs/preview/voice_anmingyuan_v3.1.mp3",
        "outputs/preview/voice_longsanshu_v3.1.mp3",
    ]
    for p in paths:
        print("=" * 60)
        print("FILE:", p)
        for m in ("qwen-audio-asr", "qwen-audio-asr-latest"):
            print(f"  [{m}] -> {transcribe(p, m)}")
            break
