"""Batch TTS with the LOCAL IndexTTS-2.5 model (no cloud quota needed).

Run this with the IndexTTS project's own venv, because all its deps live
there:

    D:/source_code/index-tts-new/.venv/Scripts/python.exe \
        tools/tts_index_batch.py --voice voice_03.wav --limit 2

The model is loaded ONCE and every line is synthesised in-process -- calling
the CLI per sentence would reload the checkpoint each time.

Reference voices (D:/source_code/index-tts-new/examples/):
    voice_03.wav  年轻女声 · 活泼      <- the playful one
    voice_09.wav  温柔女声 · 可爱
    voice_04.wav  成熟女声 · 专业
    voice_05/06/07.wav  男声

Emotion vector order: [happy, angry, sad, afraid, disgusted, melancholic,
surprised, calm]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PROJECT = r"D:\source_code\index-tts-new"
DEFAULT_VOICE = "voice_03.wav"          # 年轻女声 · 活泼
DEFAULT_EMO = "0.7,0,0,0,0,0,0.25,0"    # playful: happy + a bit of surprise


def collect(script_path: str):
    """Return [(tag, text), ...] for every narration line."""
    data = json.load(open(script_path, encoding="utf-8"))
    out = []
    segs = ([data["hook"]] if data.get("hook") else []) + data["stages"] + \
           ([data["outro"]] if data.get("outro") else [])
    for seg in segs:
        items = sorted(seg["lines"], key=lambda l: l["at"])
        for i, line in enumerate(items):
            out.append((f"idx_{seg['id']}_{i}", line["text"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=os.path.join(
        ROOT, "video14", "script_v.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "tts_index"))
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--emo-vector", default=DEFAULT_EMO)
    ap.add_argument("--emo-alpha", type=float, default=0.75)
    ap.add_argument("--lang", default="ZH")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, args.project)
    os.chdir(args.project)          # checkpoints/ + examples/ are relative

    from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

    items = collect(args.script)
    if args.limit:
        items = items[: args.limit]
    emo = [float(x) for x in args.emo_vector.split(",")] if args.emo_vector \
        else None

    print(f"loading IndexTTS-2.5 from {args.project} ...", flush=True)
    t0 = time.time()
    tts = IndexTTS2(cfg_path="checkpoints/config.yaml",
                    model_dir="checkpoints")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    for tag, text in items:
        dst = os.path.join(args.out, tag + ".wav")
        if os.path.exists(dst) and os.path.getsize(dst) > 2000:
            print(f"  skip {tag}", flush=True)
            continue
        t = time.time()
        tts.infer(spk_audio_prompt=os.path.join("examples", args.voice),
                  text=text, output_path=dst, lang=args.lang,
                  emo_vector=emo, emo_alpha=args.emo_alpha)
        try:
            w = wave.open(dst)
            dur = w.getnframes() / w.getframerate()
        except Exception:
            dur = -1
        print(f"  ok {tag}: {dur:.2f}s ({time.time() - t:.1f}s)  {text}",
              flush=True)


if __name__ == "__main__":
    main()
