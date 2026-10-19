"""Synthesise the narration for the 9:16 cut and rebuild its timeline.

The 9:16 cut is SEGMENT based: hook -> 1-2 full run -> 1-3 full run ->
1-4 full run -> outro.  Every script line carries an `at` (seconds into its
segment), so instead of concatenating sentences back to back this writes each
line INTO ITS SLOT inside the segment track -- the voice always lands on the
moment of footage it talks about.

If a line is longer than the room before the next one, it is time-stretched
with ffmpeg `atempo` (pitch preserved) rather than overlapping.

Outputs
  video14/public/audio/nar_v.wav   one continuous narration track
  outputs/tts_len_v.json           final per-line durations
  video14/src/timings_v.json       rebuilt with narration: true

    python tools/make_narration_v.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import wave

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import imageio_ffmpeg  # noqa: E402
from tts_batch import synth, load_key  # noqa: E402

SCRIPT = os.path.join(_ROOT, "video14", "script_v.json")
AUDIO_OUT = os.path.join(_ROOT, "video14", "public", "audio", "nar_v.wav")
TIMINGS = os.path.join(_ROOT, "video14", "src", "timings_v.json")
LEN_OUT = os.path.join(_ROOT, "outputs", "tts_len_v.json")
TMP = os.path.join(_ROOT, "outputs", "tts")
TTS_INDEX = os.path.join(_ROOT, "outputs", "tts_index")

SR = 48000
LEAD = 0.06
TAIL = 0.16
BREATH = 0.25          # gap kept before the next line
MAX_CPS = 11.0         # faster than this => the model skipped the tail
FF = imageio_ffmpeg.get_ffmpeg_exe()

PROMPT = "成熟的中文男声纪录片旁白，语气自然笃定，语速正常，发音清晰。"

ENGINE = "dashscope"   # or "index" (local IndexTTS-2.5, playful female)


def read_wav(path):
    """Read any PCM wav -> float32 [n, ch], normalised, plus its sample rate.

    Must handle sampwidth 1/2/4: ffmpeg happily writes 8- or 32-bit wavs and
    decoding those as int16 halves/doubles the apparent length (this silently
    corrupted the atempo round-trip)."""
    w = wave.open(path)
    n, sw, ch = w.getnframes(), w.getsampwidth(), w.getnchannels()
    raw = w.readframes(n)
    if sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) \
            / 128.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sampwidth {sw} in {path}")
    d = arr.reshape(-1, ch)
    return d, w.getframerate()


def write_wav(path, data):
    """Always write real stereo: mono arrays would otherwise be written as
    half as many frames (the bytes get re-read as 2 channels), which silently
    halved every atempo round-trip."""
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(
            (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())


def resample(d, src_sr):
    if src_sr == SR:
        return d
    n = int(round(len(d) * SR / src_sr))
    idx = np.linspace(0, len(d) - 1, n)
    out = np.stack([np.interp(idx, np.arange(len(d)), d[:, c])
                    for c in range(d.shape[1])], axis=1)
    return out


def trim(d):
    mono = d.mean(1)
    seg = int(0.02 * SR)
    if len(mono) < seg:
        return d
    n = len(mono) // seg
    db = np.array([
        20 * np.log10(max(np.sqrt((mono[i * seg:(i + 1) * seg] ** 2).mean()),
                          1e-9))
        for i in range(n)
    ])
    loud = np.where(db > -38)[0]
    if len(loud) == 0:
        return d
    s = max(0, int(loud[0] * seg - LEAD * SR))
    e = min(len(d), int((loud[-1] + 1) * seg + TAIL * SR))
    return d[s:e]


def atempo(d, factor):
    """Time-stretch with ffmpeg atempo (pitch preserved)."""
    src = os.path.join(TMP, "_atempo_in.wav")
    dst = os.path.join(TMP, "_atempo_out.wav")
    write_wav(src, d)
    if os.path.exists(dst):
        os.remove(dst)
    cmd = [FF, "-y", "-i", src, "-filter:a", f"atempo={factor:.4f}",
           "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("   atempo failed:", r.stderr[-300:])
        return d
    out, sr = read_wav(dst)
    return resample(out, sr)


def split_point(text: str) -> int:
    for p in ("，", "。", "、", "；", "—", " ", ","):
        i = text.rfind(p, 0, len(text) - 2)
        if len(text) // 3 <= i <= len(text) * 2 // 3:
            return i + 1
    return len(text) // 2


def speak(text: str, tag: str, key: str) -> np.ndarray:
    """One line of speech.  ENGINE=index reads the local IndexTTS-2.5 output,
    otherwise fall back to the DashScope qwen voice."""
    if ENGINE == "index":
        path = os.path.join(TTS_INDEX, f"idx_{tag.split('v_', 1)[-1]}.wav")
        if not os.path.exists(path):
            raise SystemExit(f"missing local TTS file: {path} (run "
                             f"tools/tts_index_batch.py first)")
        d, sr = read_wav(path)
        return trim(resample(d, sr))
    path = synth(text, PROMPT, tag, key)
    d, sr = read_wav(path)
    if sr != SR:
        d = resample(d, sr)
    d = trim(d)
    n = sum(1 for c in text if not c.isspace())
    cps = n / max(len(d) / SR, 1e-6)
    if cps > MAX_CPS and len(text) > 8:
        i = split_point(text)
        print(f"      ! {cps:.1f} 字/秒 -> split: {text[:i]} | {text[i:]}")
        a = trim(resample(*reversed(read_wav(synth(text[:i], PROMPT, tag + "a",
                                                   key)))))
        b = trim(resample(*reversed(read_wav(synth(text[i:], PROMPT, tag + "b",
                                                   key)))))
        d = np.concatenate([a, b])
    return d


def main() -> None:
    global ENGINE
    ENGINE = "index" if "--engine" in sys.argv and \
        sys.argv[sys.argv.index("--engine") + 1] == "index" else "dashscope"
    print(f"engine = {ENGINE}")
    script = json.load(open(SCRIPT, encoding="utf-8"))
    key = None if ENGINE == "index" else load_key()
    os.makedirs(TMP, exist_ok=True)

    CLIP_LEN = {"c12_player.mp4": 63.44, "c13_player.mp4": 64.36,
                "c14_player.mp4": 100.20}
    stages = script["stages"]
    segs = ([script["hook"]] if script.get("hook") else []) + stages + \
           ([script["outro"]] if script.get("outro") else [])

    tracks, lens, cursor = [], {}, 0.0
    for seg in segs:
        dur = seg.get("dur") or CLIP_LEN[seg["clip"]]
        track = np.zeros((int(round(dur * SR)), 2), dtype=np.float32)
        items = sorted(seg["lines"], key=lambda l: l["at"])
        print(f"--- {seg['id']} ({len(items)} lines, {dur:.2f}s) ---",
              flush=True)
        for i, line in enumerate(items):
            at = line["at"]
            room = (items[i + 1]["at"] - BREATH) - at if i + 1 < len(items) \
                else (dur - 0.3) - at
            a = speak(line["text"], f"v_{seg['id']}_{i}", key)
            d = len(a) / SR
            if d > room and room > 0.6:
                factor = min(1.35, d / room)
                print(f"      ~ {d:.2f}s > room {room:.2f}s -> "
                      f"atempo x{factor:.2f}")
                a = atempo(a, factor)
                d = len(a) / SR
            start = int(round(at * SR))
            end = min(len(track), start + len(a))
            if end > start:
                track[start:end] += a[: end - start]
            lens[f"{seg['id']}_{i}"] = round(d, 3)
            print(f"   @{at:5.1f}s {d:5.2f}s  {line['text']}", flush=True)
            time.sleep(0.12)
        tracks.append(track)
        cursor += dur

    merged = np.concatenate(tracks)
    write_wav(AUDIO_OUT, merged)
    json.dump(lens, open(LEN_OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"\nnarration {len(merged) / SR:.2f}s -> {AUDIO_OUT}")

    timings = json.load(open(TIMINGS, encoding="utf-8"))
    timings["narration"] = True
    json.dump(timings, open(TIMINGS, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("timings_v.json: narration = True (re-run build_v_timings.py "
          "--from-audio to refresh line durations)")


if __name__ == "__main__":
    main()
