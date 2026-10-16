"""Preview scene 1 with the FIXED parameter shape (text_prompt = script),
and verify with ASR that the model really reads our words."""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_verify import asr  # noqa: E402
from tts_batch import END, MODEL, OUT, load_key  # noqa: E402
from make_narration2 import SR, GAP, LEAD, TAIL, PROMPT, write_wav  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")
PREVIEW = os.path.join(ROOT, "outputs", "preview")
VOICE = "anmingyuan_v3.1"
TAG = sys.argv[1] if len(sys.argv) > 1 else "normal"
STYLE = {
    "normal": PROMPT,
    "slow": PROMPT + "语速舒缓从容，句子之间停顿明显，不要赶。",
}[TAG]


def synth_get_url(text, key):
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": text},
        "parameters": {"voice": VOICE, "instruction": STYLE},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        END, data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return r["output"]["audio"]["url"]


def read_wav(path):
    w = wave.open(path)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return d.reshape(-1, w.getnchannels())


def trim(audio):
    mono = audio.astype(np.float32).mean(1) / 32768.0
    seg = 960
    n = len(mono) // seg
    db = np.array([20 * np.log10(max(np.sqrt(
        (mono[i * seg:(i + 1) * seg] ** 2).mean()), 1e-9)) for i in range(n)])
    loud = np.where(db > -38)[0]
    if len(loud) == 0:
        return audio
    s = max(0, int(loud[0] * seg - LEAD * SR))
    e = min(len(audio), int((loud[-1] + 1) * seg + TAIL * SR))
    return audio[s:e]


if __name__ == "__main__":
    os.makedirs(PREVIEW, exist_ok=True)
    key = load_key()
    script = json.load(open(SCRIPT, encoding="utf-8"))
    sc = script[0]
    pieces = []
    print("voice:", VOICE)
    for i, line in enumerate(sc["lines"]):
        url = synth_get_url(line, key)
        heard = asr(url)
        tmp = os.path.join(OUT, f"fix_{TAG}_{sc['name']}_{i}.wav")
        urllib.request.urlretrieve(url, tmp)
        a = trim(read_wav(tmp))
        dur = len(a) / SR
        print(f"  [{i}] {len(line):2d}字 {dur:5.2f}s  {len(line)/dur:5.2f}字/秒")
        print(f"      ASK : {line}")
        print(f"      HEARD: {heard}")
        pieces.append(a)
        pieces.append(np.zeros((int(GAP * SR), 2), dtype=np.int16))

    merged = np.concatenate(pieces)
    wav = os.path.join(PREVIEW, f"s01_{TAG}.wav")
    write_wav(wav, merged)
    mp3 = os.path.join(PREVIEW, f"s01_{TAG}.mp3")
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", wav,
                    "-c:a", "libmp3lame", "-b:a", "192k", mp3],
                   capture_output=True)
    print(f"\n=> {mp3}  {len(merged)/SR:.2f}s  {os.path.getsize(mp3)//1024}KB")
