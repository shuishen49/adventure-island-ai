"""Rebuild the narration.

Why this exists: qwen-audio-3.1-tts-next only reads ~15 characters
reliably per request. Longer text gets compressed / partially skipped
(measured: 46 chars -> 3 s = 15 chars/sec). So every line is synthesised
separately, trimmed, and concatenated.

Outputs:
  video/public/audio/sNN.wav   (standard RIFF header, safe for Remotion)
  video/src/timings.json       (per-scene + per-line timings for subtitles)
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import synth, load_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")
AUDIO_DIR = os.path.join(ROOT, "video", "public", "audio")

# Identical prompt for every line -> stable timbre across the whole video.
PROMPT = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常，发音清晰。"

SR = 48000
GAP = 0.14          # silence between sentences
LEAD = 0.10         # keep a little air before speech starts
TAIL = 0.24         # and after it ends
SCENE_GAP = 0.35    # between scenes


def read_wav(path):
    w = wave.open(path)
    n = w.getnframes()
    d = np.frombuffer(w.readframes(n), dtype=np.int16)
    d = d.reshape(-1, w.getnchannels())
    return d


def trim(audio):
    """Cut leading/trailing silence based on an RMS envelope."""
    mono = audio.astype(np.float32).mean(1) / 32768.0
    seg = 960  # 0.02 s
    if len(mono) < seg:
        return audio
    n = len(mono) // seg
    db = np.array([
        20 * np.log10(max(np.sqrt((mono[i * seg:(i + 1) * seg] ** 2).mean()), 1e-9))
        for i in range(n)
    ])
    loud = np.where(db > -38)[0]
    if len(loud) == 0:
        return audio
    s = max(0, int((loud[0] * seg) - LEAD * SR))
    e = min(len(audio), int(((loud[-1] + 1) * seg) + TAIL * SR))
    return audio[s:e]


def write_wav(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.astype(np.int16).tobytes())


def main():
    script = json.load(open(SCRIPT, encoding="utf-8"))
    key = load_key()
    os.makedirs(AUDIO_DIR, exist_ok=True)

    scenes = []
    cursor = 0.0
    fps = 30

    for sc in script:
        pieces = []
        lines = []
        pos = 0.0
        print(f"--- {sc['id']} ({len(sc['lines'])} lines) ---")
        for i, line in enumerate(sc["lines"]):
            tag = f"{sc['name']}_{i}"
            raw = synth(line, PROMPT, tag, key)
            a = trim(read_wav(raw))
            dur = len(a) / SR
            lines.append({"text": line, "start": round(pos, 3),
                          "dur": round(dur, 3)})
            pos += dur + GAP
            pieces.append(a)
            gap = np.zeros((int(GAP * SR), 2), dtype=np.int16)
            pieces.append(gap)
            print(f"   {len(line):2d}字 {dur:5.2f}s  {len(line) / dur:5.2f} 字/秒 | {line}")
            time.sleep(0.2)

        merged = np.concatenate(pieces) if pieces else np.zeros((1, 2), np.int16)
        out = os.path.join(AUDIO_DIR, sc["name"] + ".wav")
        write_wav(out, merged)
        total = len(merged) / SR
        chars = sum(len(x) for x in sc["lines"])
        print(f"   => {sc['id']} total {total:.2f}s  ({chars} 字, "
              f"{chars / total:.2f} 字/秒)")

        scenes.append({
            "id": sc["id"],
            "name": sc["name"],
            "audio": "audio/" + sc["name"] + ".wav",
            "title": sc["title"],
            "sub": sc["sub"],
            "kind": sc["kind"],
            "visual": sc["visual"],
            "dur": round(total, 3),
            "durFrames": int(round(total * fps)),
            "startFrames": int(round(cursor * fps)),
            "lines": lines,
        })
        cursor += total + SCENE_GAP

    total_frames = int(round(cursor * fps))
    json.dump({"fps": fps, "durationInFrames": total_frames, "scenes": scenes},
              open(os.path.join(ROOT, "video", "src", "timings.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nTOTAL {cursor:.2f}s / {total_frames} frames")


if __name__ == "__main__":
    main()
