"""Final narration builder.

What changed vs. every earlier attempt:
  * LONG-TEXT synthesis works (44 chars -> 7.6 s, 529 chars -> 60 s).
    The old "only ~15 chars" rule was an artifact of the wrong parameter
    shape, not a model limit. Single-request cap is ~60 s of audio.
  * `input.text_prompt` is the spoken text. Putting the BGM description in
    the SAME field (not in `instruction`) makes the model emit voice + music
    together -- verified: noise floor stays at -16..-30 dB with no silent
    gaps, and ASR still transcribes the script verbatim.
  * Magnetic male voice: longsanshu_v3.1 (沉稳质感男).
  * Subtitle timing comes from ASR sentence timestamps, interpolated by
    character count, so captions follow the audio even though the audio is
    synthesised as whole paragraphs.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import END, MODEL, load_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")
AUDIO_DIR = os.path.join(ROOT, "video", "public", "audio")
PREVIEW = os.path.join(ROOT, "outputs", "preview")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(PREVIEW, exist_ok=True)

VOICE = os.environ.get("VOICE", "longsanshu_v3.1")
STYLE = "成熟的中文男声纪录片旁白，声音磁性低沉有质感，语气笃定从容，发音清晰。"
BGM_NOTE = ("\n（背景音乐：史诗感的管弦乐，弦乐铺底加鼓点推进，"
            "音量压低，不要盖过人声。）")

ASR_SUBMIT = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
ASR_TASK = "https://dashscope.aliyuncs.com/api/v1/tasks/"

FPS = 30
TAIL = 0.45     # keep a little of the music bed after the last sentence


def synth(text, name):
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": text + BGM_NOTE},
        "parameters": {"voice": VOICE, "instruction": STYLE},
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


def asr_sentences(url, key):
    body = json.dumps({"model": "paraformer-v2",
                       "input": {"file_urls": [url]}}).encode()
    req = urllib.request.Request(
        ASR_SUBMIT, data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "X-DashScope-Async": "enable"})
    tid = json.load(urllib.request.urlopen(req, timeout=120))["output"]["task_id"]
    for _ in range(60):
        time.sleep(3)
        s = json.load(urllib.request.urlopen(
            urllib.request.Request(ASR_TASK + tid,
                                   headers={"Authorization": "Bearer " + key}),
            timeout=60))
        if s["output"]["task_status"] == "SUCCEEDED":
            break
    else:
        return []
    tr = json.load(urllib.request.urlopen(
        s["output"]["results"][0]["transcription_url"], timeout=60))["transcripts"][0]
    return [(x["begin_time"] / 1000.0, x["end_time"] / 1000.0, x["text"])
            for x in tr.get("sentences", [])]


def map_lines(lines, sents, part_dur, char_offset=0):
    """Distribute sentence timings over our script lines by char ratio.

    `char_offset` = how many characters of the part's text were already
    consumed by earlier scenes, so timings keep advancing across scenes.
    """
    out = []
    if not sents:
        d = part_dur / max(1, len(lines))
        pos = 0.0
        for ln in lines:
            out.append({"text": ln, "start": round(pos, 3),
                        "dur": round(d, 3)})
            pos += d
        return out, part_dur
    cursor_asr = char_offset
    for ln in lines:
        need = len(ln)
        # start: sentence covering the current char position
        acc, start = 0, None
        for b, e, t in sents:
            if acc + len(t) > cursor_asr:
                frac = (cursor_asr - acc) / max(1, len(t))
                start = b + (e - b) * frac
                break
            acc += len(t)
        if start is None:
            start = sents[-1][1]
        cursor_asr += need
        acc, end = 0, None
        for b, e, t in sents:
            if acc + len(t) >= cursor_asr:
                frac = (cursor_asr - acc) / max(1, len(t))
                end = b + (e - b) * frac
                break
            acc += len(t)
        if end is None:
            end = sents[-1][1]
        out.append({"text": ln,
                    "start": round(start, 3),
                    "dur": round(max(0.4, end - start), 3)})
    last = max(x["start"] + x["dur"] for x in out)
    return out, min(part_dur, last + TAIL)


if __name__ == "__main__":
    KEY = load_key()
    script = json.load(open(SCRIPT, encoding="utf-8"))

    # two parts keeps each request well under the ~60 s cap
    groups = [script[:5], script[5:]]
    scenes = []
    cursor_frames = 0
    cache_path = os.path.join(ROOT, "video", "src", "parts_cache.json")
    cache = json.load(open(cache_path, encoding="utf-8")) \
        if os.path.exists(cache_path) else {}
    for gi, g in enumerate(groups):
        text = "".join("".join(s["lines"]) for s in g)
        name = f"part{gi + 1}"
        print(f"=== {name}: {g[0]['id']}..{g[-1]['id']}  {len(text)} 字 ===")
        dst = os.path.join(AUDIO_DIR, name + ".wav")
        if name in cache and os.path.exists(dst):
            dur, sents = cache[name]["dur"], cache[name]["sents"]
            print(f"    cached audio {dur}s, {len(sents)} sentences")
        else:
            path, dur, url = synth(text, name)
            print(f"    audio {dur}s -> {path}")
            sents = asr_sentences(url, KEY)
            print(f"    ASR {len(sents)} sentences")
            cache[name] = {"dur": dur, "sents": sents, "url": url}
            json.dump(cache, open(cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False)

        # walk scenes inside this part
        char_off = 0
        first = True
        for sc in g:
            lines, sc_dur_end = map_lines(sc["lines"], sents, dur, char_off)
            sc_start = 0.0 if first else max(0.0, lines[0]["start"] - 0.15)
            sc_dur = max(1.0, sc_dur_end - sc_start)
            frames = int(round(sc_dur * FPS))
            # lines are stored relative to the scene start
            rel = [{"text": x["text"],
                    "start": round(x["start"] - sc_start, 3),
                    "dur": x["dur"]} for x in lines]
            scenes.append({
                "id": sc["id"], "name": sc["name"],
                "audio": f"audio/{name}.wav",
                "audioOffset": round(sc_start, 3),
                "title": sc["title"], "sub": sc["sub"],
                "kind": sc["kind"], "visual": sc["visual"],
                "dur": round(sc_dur, 3),
                "durFrames": frames,
                "startFrames": cursor_frames,
                "lines": rel,
            })
            print(f"    {sc['id']} start={sc_start:6.2f}s dur={sc_dur:6.2f}s "
                  f"({frames}f)  首句 {rel[0]['start']:+.2f}s")
            cursor_frames += frames
            char_off += sum(len(x) for x in sc["lines"])
            first = False

    total_frames = cursor_frames
    json.dump({"fps": FPS, "durationInFrames": total_frames, "scenes": scenes},
              open(os.path.join(ROOT, "video", "src", "timings.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nTOTAL {total_frames} frames = {total_frames / FPS:.2f}s")

    # full-length preview for the user
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    lst = os.path.join(PREVIEW, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for gi in range(len(groups)):
            f.write(f"file '{os.path.join(AUDIO_DIR, f'part{gi+1}.wav')}'\n")
    full = os.path.join(PREVIEW, "full_narration.mp3")
    subprocess.run([exe, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:a", "libmp3lame", "-b:a", "192k", full],
                   capture_output=True)
    os.remove(lst)
    print("preview:", full, os.path.getsize(full) // 1024, "KB")
