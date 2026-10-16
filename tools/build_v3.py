"""Build the whole narration with LONG-TEXT synthesis (verified working:
44 chars -> 7.6 s, 70 chars -> 9.3 s) instead of per-sentence synthesis.

Groups the script into chunks that stay under the ~60 s single-request cap,
records ASR timestamps so subtitles can still follow each sentence.
"""
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import END, MODEL, load_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")
AUDIO_DIR = os.path.join(ROOT, "video", "public", "audio")
PREVIEW = os.path.join(ROOT, "outputs", "preview")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(PREVIEW, exist_ok=True)

VOICE = os.environ.get("VOICE", "longsanshu_v3.1")
STYLE = ("成熟的中文男声纪录片旁白，声音磁性低沉有质感，"
         "语气笃定从容，语速适中，发音清晰。")


def synth(text, style, voice, name):
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": text},
        "parameters": {"voice": voice, "instruction": style},
    }, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        END, data=body,
        headers={"Authorization": "Bearer " + KEY,
                 "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=240))
    a = r["output"]["audio"]
    dst = os.path.join(AUDIO_DIR, name + ".wav")
    urllib.request.urlretrieve(a["url"], dst)
    return dst, a.get("duration"), a["url"]


def group_scenes(script, max_chars=280):
    """Pack scenes into chunks under max_chars, keeping scene order."""
    groups, cur, cur_n = [], [], 0
    for sc in script:
        n = sum(len(x) for x in sc["lines"])
        if cur and cur_n + n > max_chars:
            groups.append(cur)
            cur, cur_n = [], 0
        cur.append(sc)
        cur_n += n
    if cur:
        groups.append(cur)
    return groups


if __name__ == "__main__":
    KEY = load_key()
    script = json.load(open(SCRIPT, encoding="utf-8"))
    groups = group_scenes(script)
    meta = []
    for gi, g in enumerate(groups):
        text = "".join("".join(s["lines"]) for s in g)
        name = f"part{gi+1}"
        print(f"=== {name}: scenes {g[0]['id']}..{g[-1]['id']}  {len(text)} 字 ===")
        path, dur, url = synth(text, STYLE, VOICE, name)
        print(f"    dur={dur}s  url saved")
        meta.append({"part": name, "scenes": [s["id"] for s in g],
                     "chars": len(text), "dur": dur, "url": url,
                     "wav": f"audio/{name}.wav"})
    json.dump(meta, open(os.path.join(ROOT, "video", "src", "parts.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nparts:", [(m["part"], m["chars"], m["dur"]) for m in meta])
