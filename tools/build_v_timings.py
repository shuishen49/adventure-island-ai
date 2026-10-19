"""Build src/timings_v.json for the 9:16 cut.

Layout is SEGMENTS, not scenes:  hook -> 1-2 full run -> 1-3 full run ->
1-4 full run -> outro.  Each run is played end to end and every narration
line is pinned to the second it belongs to inside that run, so a line about
1-2 is always spoken over 1-2 footage.

Two timing sources:
  * default -- estimate from character count (used before the TTS exists)
  * --from-audio -- measure the real TTS wavs (see tools/make_narration_v.py)

    python tools/build_v_timings.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "video14", "script_v.json")
OUT = os.path.join(ROOT, "video14", "src", "timings_v.json")

FPS = 30

# real lengths of the source runs (ffprobe)
CLIP_LEN = {
    "c12_player.mp4": 63.44,
    "c13_player.mp4": 64.36,
    "c14_player.mp4": 100.20,
}

CHARS_PER_SEC = 5.5
MIN_LINE = 2.0
BREATH = 0.25          # never let a line run into the next one


def estimate(text: str) -> float:
    n = sum(1 for c in text if not c.isspace())
    return max(MIN_LINE, n / CHARS_PER_SEC + 0.4)


def pack(seg: dict, dur: float, measured: dict | None) -> list[dict]:
    """measured: {(seg_id, line_index): seconds} from real TTS, if any."""
    items = sorted(seg["lines"], key=lambda l: l["at"])
    out = []
    for i, line in enumerate(items):
        start = line["at"]
        want = (measured or {}).get((seg["id"], i)) or estimate(line["text"])
        room = (items[i + 1]["at"] - BREATH) - start if i + 1 < len(items) \
            else (dur - 0.3) - start
        d = min(want, max(1.2, room))
        item = {"text": line["text"], "start": round(start, 3),
                "dur": round(d, 3)}
        if line.get("ai"):
            item["ai"] = True       # pop the model's own view while spoken
        out.append(item)
    return out


def main() -> None:
    measured = None
    if "--from-audio" in sys.argv:
        path = os.path.join(ROOT, "outputs", "tts_len_v.json")
        raw = json.load(open(path, encoding="utf-8"))
        measured = {}
        for k, v in raw.items():
            sid, idx = k.rsplit("_", 1)
            measured[(sid, int(idx))] = float(v)
        print(f"using measured TTS lengths for {len(measured)} lines")

    script = json.load(open(SRC, encoding="utf-8"))
    stages = script["stages"]
    segs = []
    if script.get("hook"):
        segs.append(script["hook"])
    segs.extend(stages)
    if script.get("outro"):
        segs.append(script["outro"])

    out, cursor = [], 0.0
    for seg in segs:
        dur = seg.get("dur") or CLIP_LEN[seg["clip"]]
        out.append({
            "id": seg["id"],
            "kind": "stage" if seg in stages else seg["id"],
            "clip": seg["clip"],
            # same rollout, same frame, model's view -- kept in sync by
            # starting it at the very same offset
            "clipAi": seg["clip"].replace("_player", "_agent"),
            "clipStart": seg.get("clipStart", 0),
            "label": seg.get("label"),
            "title": seg.get("title", ""),
            "sub": seg.get("sub", ""),
            "startSec": round(cursor, 3),
            "durSec": round(dur, 3),
            "startFrames": int(round(cursor * FPS)),
            "durFrames": int(round(dur * FPS)),
            "lines": pack(seg, dur, measured),
        })
        cursor += dur

    total = int(round(cursor * FPS))
    audio = os.path.join(ROOT, "video14", "public", "audio", "nar_v.wav")
    json.dump({"fps": FPS, "durationInFrames": total,
               "narration": os.path.exists(audio), "segments": out},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(out)} segments, {total} frames = {cursor:.1f}s -> {OUT}"
          f"   narration={os.path.exists(audio)}")
    for s in out:
        print(f"  {s['id']:6s} {s['durSec']:6.2f}s "
              f"@[{s['startSec']:6.2f}]  {s['title']}  "
              f"clip={s['clip']} from {s['clipStart']}s")


if __name__ == "__main__":
    main()
