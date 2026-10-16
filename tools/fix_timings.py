"""Rebuild video/src/timings.json using precise mp3 durations."""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = imageio_ffmpeg.get_ffmpeg_exe()


def dur(path):
    r = subprocess.run([EXE, "-i", path, "-hide_banner"],
                       capture_output=True, text=True)
    txt = (r.stderr or "") + (r.stdout or "")
    m = re.search(r"Duration:\s+(\d+):(\d+):([\d.]+)", txt)
    if not m:
        raise RuntimeError("no duration for " + path)
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def main():
    t = json.load(open(os.path.join(ROOT, "video", "timings_raw.json"),
                       encoding="utf-8"))
    fps = 30
    out = []
    cursor = 0.0
    for it in t:
        name = os.path.basename(it["audio"]).replace(".wav", ".mp3")
        p = os.path.join(ROOT, "video", "public", "audio", name)
        d = round(dur(p), 3)
        out.append({**it, "audio": "audio/" + name, "dur": d,
                    "durFrames": int(round(d * fps)),
                    "startFrames": int(round(cursor * fps))})
        cursor += d + 0.25
    total = int(round(cursor * fps))
    json.dump({"fps": fps, "durationInFrames": total, "scenes": out},
              open(os.path.join(ROOT, "video", "src", "timings.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print("total", round(cursor, 2), "sec /", total, "frames")
    for s in out:
        print(" ", s["id"], s["dur"], "start", s["startFrames"])


if __name__ == "__main__":
    main()
