"""build_v6.py — narration rebuild v2 for the Adventure Island video.

Implements the plan in outputs/v5/优化方案_v2.md:

  1. TTS   same voice + instruction as v5, RATE 0.5 -> 0.7.
           Measured on the same 40 chars: 0.5 = 211 字/分 (the API's slowest
           setting; it drags the vowels, which is the "sounds like slow
           playback" complaint), 0.6 = 255, 0.7 = 282, 1.0 = 426.
  1b. pitch consistency by RE-ROLLING, not by shifting.
           v5 rolled part2 at 152.5 Hz and forced it down 4.18 semitones; on the
           same take that cost 20% of the detected syllable onsets and 0.5pt of
           the 2-8 kHz energy. v6 rolls part2 until the model hands back a take
           near part1's pitch and then corrects at most MAX_RESID_ST semitones.
  2. pauses: silent runs longer than TRIM_MIN are cut back to TRIM_TO (v5 left
           24% of the runtime silent, 15 pauses over 0.4s, 4.5s between parts).
  3. timings: ASR WORD timestamps (parameters.enable_words, verified available)
           + a content alignment (difflib over a numeral/latin-normalised char
           stream) replace v5's char-count ratio over sentence timestamps --
           that ratio is why every scene in v5 ran 1.7-7.4s late.
  4. mix: unchanged from v5 (BGM bed, side-chain ducking, limiter, fade out).

Usage:  python tools/build_v6.py [--force]
        RATE / TRIM_MIN / TRIM_TO / PART_GAP env vars override the knobs.
        Rolls and ASR results are cached in outputs/v6/v6_cache.json, so the
        timing can be re-tuned without new API calls.
"""
import bisect
import difflib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
import glob

import numpy as np
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import END, MODEL, load_key  # noqa: E402
import make_bgm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")
PUBLIC_AUDIO = os.path.join(ROOT, "video", "public", "audio")
TIMINGS = os.path.join(ROOT, "video", "src", "timings.json")
V6 = os.path.join(ROOT, "outputs", "v6")
ROLLS = os.path.join(V6, "rolls")
CACHE = os.path.join(V6, "v6_cache.json")

VOICE = os.environ.get("VOICE", "longsanshu_v3.1")
STYLE = (
    "成熟的中文男声纪录片旁白，声音磁性低沉有质感，情绪饱满富有感染力，"
    "语气笃定从容、张弛有度，像讲述一段亲身经历的技术故事，节奏自然，发音清晰。"
    "全篇保持同一个低沉稳定的音色和音高，不要因为情绪起伏抬高音调，"
    "整体像一个人在同一个录音棚里一口气讲完。"
)

RATE = float(os.environ.get("RATE", "0.7"))

# Narration pitch. v5's clamp was 105-125 Hz and it forced part2 down 4.18
# semitones to get there. v6 keeps the same deep register but reaches it by
# re-rolling; the target is only a preference, the accept window is where a take
# is used as-is.
F0_TARGET = 118.0
F0_ACCEPT = (110.0, 126.0)
F0_CLAMP = (105.0, 125.0)
MAX_RESID_ST = float(os.environ.get("MAX_RESID_ST", "1.5"))
ROLLS_MAX = int(os.environ.get("ROLLS_MAX", "6"))

SR = 48000
FPS = 30
LEAD = 0.15          # scene starts this much before its first line
TAIL = 0.5           # last scene keeps this much after its last line
PART_GAP = float(os.environ.get("PART_GAP", "0.5"))
BGM_GAP_DB = -8.0
DUCK_DB = 13.0
WIN = 480            # 10 ms envelope block

TRIM_MIN = float(os.environ.get("TRIM_MIN", "0.50"))   # only pauses longer than this
TRIM_TO = float(os.environ.get("TRIM_TO", "0.34"))     # are cut back to this
HEAD_TO = float(os.environ.get("HEAD_TO", "0.15"))
TAIL_TO = 0.40
PAUSE_GATE_DB = -22.0

ASR_SUBMIT = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
ASR_TASK = "https://dashscope.aliyuncs.com/api/v1/tasks/"


# ---------------------------------------------------------------- wav helpers
def read_wav_f(path):
    w = wave.open(path)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return d.reshape(-1, w.getnchannels()).astype(np.float32) / 32768.0


def write_wav_f(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())


def dur_of(path):
    """Duration of a raw API WAV (its header carries a bogus data size)."""
    w = wave.open(path)
    ch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
    w.close()
    return max(0.0, (os.path.getsize(path) - 44) / float(fr * ch * sw))


def rms_db(a):
    mono = a.mean(1)
    return 20 * math.log10(max(float(np.sqrt((mono ** 2).mean())), 1e-9))


def speech_rms_db(a):
    mono = a.mean(1)
    n = len(mono) // WIN
    if n == 0:
        return rms_db(a)
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    loud = db[db > -45.0]
    return float(loud.mean()) if len(loud) else rms_db(a)


def loud_db(a):
    """Gated loudness: energy-domain mean of 10 ms blocks above -45 dB.

    speech_rms_db averages dB values, so every block counts the same no matter
    how loud it is -- a take with more low-level material (breaths, soft word
    tails) reads several dB quieter than it sounds, and matching on it left a
    2.9 dB step between the two takes. Weighting each block by its energy is
    what tracks loudness. Used to match the narration takes to each other; the
    bed level and duck gate keep the older ruler they were auditioned with.
    """
    mono = a.mean(1)
    n = len(mono) // WIN
    if n == 0:
        return rms_db(a)
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    loud = db[db > -45.0]
    if not len(loud):
        return rms_db(a)
    return float(10 * np.log10(np.mean(10 ** (loud / 10.0))))


def block_db(a):
    """Per 10 ms block level, mono."""
    mono = a.mean(1)
    n = len(mono) // WIN
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(e, 1e-9))


def measure_f0(path):
    """Median F0 over voiced frames (Praat). v5 established this as the ruler:
    it reads a synthetic 150 Hz tone as 150.0 and a known -3 semitone rubberband
    shift as -2.97."""
    try:
        import parselmouth
    except ImportError:
        return 0.0
    snd = parselmouth.Sound(read_wav_f(path).mean(1).astype(np.float64),
                            sampling_frequency=SR)
    v = snd.to_pitch(time_step=0.01, pitch_floor=75.0,
                     pitch_ceiling=400.0).selected_array["frequency"]
    v = v[v > 0]
    return float(np.median(v)) if len(v) else 0.0


def pitch_shift(src, dst, ratio):
    """Time-preserving pitch shift (ffmpeg rubberband, formants preserved)."""
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-af", f"rubberband=pitch={ratio:.6f}:formant=preserved",
                    "-c:a", "pcm_s16le", dst], check=True)


# ------------------------------------------------------------------ TTS rolls
def synth_roll(text, key, tag):
    dst = os.path.join(ROLLS, tag + ".wav")
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": text},
        "parameters": {"voice": VOICE, "instruction": STYLE, "rate": RATE},
    }, ensure_ascii=False).encode("utf-8")
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                END, data=body,
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            a = r["output"]["audio"]
            os.makedirs(ROLLS, exist_ok=True)
            urllib.request.urlretrieve(a["url"], dst)
            return dst, a["url"], float(a.get("duration") or dur_of(dst))
        except Exception as e:
            print(f"    retry {attempt + 1} {tag}: {e}")
            time.sleep(4)
    raise RuntimeError("synth failed: " + tag)


def pick_roll(text, name, key, cache, target=None):
    """Roll the part until its pitch is usable; return (wav, url, f0, dur).

    With no target (part1) the take is accepted as soon as it lands in the
    deep register we want; otherwise it is scored by distance to `target`
    (part1's pitch) and the closest take within MAX_RESID_ST wins, so the
    residual correction stays small.
    """
    want_sha = hashlib_of(text, target)
    if cache.get("parts", {}).get(name, {}).get("sha") == want_sha:
        pc = cache["parts"][name]
        if os.path.exists(pc["wav"]):
            print(f"  {name}: cached roll {os.path.basename(pc['wav'])} "
                  f"F0 {pc['f0']:.1f} Hz {pc['dur']:.2f}s")
            return pc["wav"], pc["url"], pc["f0"], pc["dur"]

    best = None
    for i in range(ROLLS_MAX):
        tag = f"{name}_r{i}"
        wav, url, dur = synth_roll(text, key, tag)
        f0 = measure_f0(wav)
        rate = len(text) / dur
        st = None if target is None else 12 * math.log2(f0 / target)
        print(f"    roll {tag}: {dur:6.2f}s  {rate * 60:5.0f} 字/分  F0 {f0:5.1f} Hz"
              + ("" if st is None else f"  ({st:+.2f} st vs part1)"))
        if not f0:
            continue
        score = abs(f0 - (target if target else F0_TARGET))
        if best is None or score < best[3]:
            best = (wav, url, f0, score, dur)
        if target is None:
            if F0_ACCEPT[0] <= f0 <= F0_ACCEPT[1]:
                break
        elif abs(st) <= 0.65:
            break
    if best is None:
        raise RuntimeError("no usable take for " + name)
    wav, url, f0, score, dur = best
    st = None if target is None else 12 * math.log2(f0 / target)
    if st is not None and abs(st) > MAX_RESID_ST:
        print(f"    !! best take is still {st:+.2f} st off; the correction will "
              f"be clamped to {MAX_RESID_ST} st")
    print(f"    -> {name} uses {os.path.basename(wav)} (F0 {f0:.1f} Hz)")
    cache.setdefault("parts", {})[name] = {
        "sha": want_sha, "wav": wav, "url": url, "f0": f0, "dur": dur,
        "rate": RATE, "voice": VOICE}
    return wav, url, f0, dur


def hashlib_of(text, target):
    import hashlib
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    h.update(f"|{VOICE}|{RATE}|{STYLE}|{target}".encode("utf-8"))
    return h.hexdigest()[:16]


def align_pitch(p1, p2, f1, f2):
    """Bring both parts to one pitch, with only a small correction allowed."""
    target = min(max(f1, F0_CLAMP[0]), F0_CLAMP[1])
    print(f"  part1 {f1:.1f} Hz -> target {target:.1f} Hz")
    out = {}
    for name, src, f0 in (("part1", p1, f1), ("part2", p2, f2)):
        dst = os.path.join(ROLLS, f"{os.path.basename(src)[:-4]}_aligned.wav")
        semis = 12 * math.log2(target / f0) if f0 else 0.0
        if abs(semis) < 0.25:
            print(f"  {name}: F0 {f0:.1f} Hz already on target, left untouched")
            out[name] = src
            continue
        if abs(semis) > MAX_RESID_ST:
            print(f"  {name}: correction {semis:+.2f} st clamped to "
                  f"{math.copysign(MAX_RESID_ST, semis):+.2f} st")
            semis = math.copysign(MAX_RESID_ST, semis)
        ratio = 2 ** (semis / 12)
        for attempt in range(3):
            pitch_shift(src, dst, ratio)
            got = measure_f0(dst)
            resid = 12 * math.log2(target / got) if got else 0.0
            if abs(resid) <= 0.35:
                break
            print(f"  {name}: pass {attempt + 1} landed at {got:.1f} Hz, "
                  f"adjusting {resid:+.2f} st")
            ratio *= 2 ** (resid / 12)
        d0, d1 = dur_of(src), dur_of(dst)
        print(f"  {name}: F0 {f0:.1f} -> {got:.1f} Hz "
              f"({12 * math.log2(got / f0):+.2f} st, formants preserved, "
              f"{d0:.2f}s -> {d1:.2f}s)")
        if abs(d0 - d1) > 0.05:
            print(f"  !! {name}: the shift changed the duration; timings are off")
        out[name] = dst
    gap = 12 * math.log2(measure_f0(out["part2"]) / measure_f0(out["part1"]))
    print(f"  aligned gap part1->part2: {gap:+.2f} st")
    return out


# ------------------------------------------------------------------- ASR words
def asr_words(url, key):
    body = json.dumps({"model": "paraformer-v2",
                       "input": {"file_urls": [url]},
                       "parameters": {"enable_words": True}}).encode()
    req = urllib.request.Request(
        ASR_SUBMIT, data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "X-DashScope-Async": "enable"})
    tid = json.load(urllib.request.urlopen(req, timeout=120))["output"]["task_id"]
    s = None
    for _ in range(80):
        time.sleep(3)
        s = json.load(urllib.request.urlopen(
            urllib.request.Request(ASR_TASK + tid,
                                   headers={"Authorization": "Bearer " + key}),
            timeout=60))
        if s["output"]["task_status"] in ("SUCCEEDED", "FAILED"):
            break
    if not s or s["output"]["task_status"] != "SUCCEEDED":
        raise RuntimeError("ASR failed: " + str(s and s["output"]["task_status"]))
    tr = json.load(urllib.request.urlopen(
        s["output"]["results"][0]["transcription_url"], timeout=60))["transcripts"][0]
    words = []
    for sent in tr.get("sentences", []):
        for w in sent.get("words") or []:
            t = (w.get("text") or "").replace(" ", "").strip()
            if t:
                words.append([w["begin_time"] / 1000.0, w["end_time"] / 1000.0, t])
    heard = "".join(x[2] for x in words)
    return words, heard


def norm_ch(c):
    """Char class for the alignment stream.

    Numerals and latin runs collapse to '#': the ASR spells numbers and product
    names differently from the script ("66号" vs "六十六号对象", "WorkBuddy" vs
    "work bud立"), and on v5's raw char counts that mismatch is what accumulated
    into the 1.7-7.4s scene drift.
    """
    if c.isdigit() or c in "一二三四五六七八九十百千万零两":
        return "#"
    c = c.lower()
    if c.isascii() and c.isalnum():
        return "#"
    return c if c.isalnum() else ""


def char_times(words):
    """Flatten ASR words into (norm_char, begin, end) at character resolution."""
    out = []
    for b, e, t in words:
        cs = [norm_ch(c) for c in t]
        cs = [c for c in cs if c]
        if not cs:
            continue
        n = len(cs)
        for i, nc in enumerate(cs):
            out.append((nc, b + (e - b) * i / n, b + (e - b) * (i + 1) / n))
    return out


def line_times(script_lines, words):
    """Content-align the script lines onto ASR char times.

    Returns ([(start, end) per line], match_pct, script_chars, asr_chars).
    """
    scrib = []
    for li, ln in enumerate(script_lines):
        for c in ln:
            nc = norm_ch(c)
            if nc:
                scrib.append((nc, li))
    asrc = char_times(words)
    A = "".join(x[0] for x in scrib)
    B = "".join(x[0] for x in asrc)
    if not A or not B:
        return [None] * len(script_lines), 0.0, len(A), len(B)
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    m = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            m[blk.a + k] = blk.b + k
    keys = sorted(m)

    def times(aidx):
        j = bisect.bisect_left(keys, aidx)
        cands = keys[max(0, j - 1): j + 1]
        k = min(cands, key=lambda x: abs(x - aidx))
        b = min(max(m[k] + (aidx - k), 0), len(asrc) - 1)
        return asrc[b][1], asrc[b][2]

    idx_by_line = {}
    for i, (_n, li) in enumerate(scrib):
        idx_by_line.setdefault(li, []).append(i)
    out = []
    for li in range(len(script_lines)):
        idxs = idx_by_line.get(li)
        if not idxs:
            out.append(None)
            continue
        a0 = times(idxs[0])[0]
        a1 = times(idxs[-1])[1]
        out.append((a0, a1))
    return out, 100.0 * len(m) / max(1, len(A)), len(A), len(B)


# --------------------------------------------------------------- pause trim
def trim_plan(a, speech_db):
    """Silent runs to shorten. Returns [(start_sample, end_sample)]."""
    db = block_db(a)
    voiced = db > (speech_db + PAUSE_GATE_DB)
    nb = len(voiced)
    runs = []
    i = 0
    while i < nb:
        if not voiced[i]:
            j = i
            while j < nb and not voiced[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    cuts = []
    for i0, j0 in runs:
        s, e = i0 * WIN, min(j0 * WIN, len(a))
        lead = (i0 == 0)
        tail = (j0 >= nb)
        keep = HEAD_TO if lead else (TAIL_TO if tail else TRIM_TO)
        if (e - s) / SR <= keep + 0.12:
            continue
        # cut the excess out of the middle: keep the decay and the pre-onset
        drop = int((e - s) - keep * SR)
        mid = (s + e) // 2
        cuts.append((mid - drop // 2, mid - drop // 2 + drop))
    return cuts, voiced


def cut_map(cuts):
    """original seconds -> trimmed seconds."""
    def f(t):
        s = 0.0
        for a, b in cuts:
            if t * SR <= a:
                break
            s += (min(t * SR, b) - a) / SR
        return t - s
    return f


def apply_trim(a, cuts):
    """Cut the intervals out, crossfading each seam (20 ms)."""
    segs, prev = [], 0
    for s, e in cuts:
        segs.append(a[prev:s])
        prev = e
    segs.append(a[prev:])
    acc = segs[0].copy()
    n = int(0.02 * SR)
    for sg in segs[1:]:
        if len(sg) == 0:
            continue
        m = min(n, len(acc), len(sg))
        if m < 8:
            acc = np.concatenate([acc, sg])
            continue
        seam = acc[-m:] * np.linspace(1, 0, m)[:, None] + sg[:m] * np.linspace(0, 1, m)[:, None]
        acc = np.concatenate([acc[:-m], seam, sg[m:]])
    return acc


# ------------------------------------------------------------------ BGM / mix
def soft_limit(a, thr=0.75, ceil=0.95):
    """Soft knee: transparent below thr, asymptotic to ceil, C1 continuous.

    Used for the loudness match. The provider's takes arrive at full scale, so
    matching the two halves by RMS drives one half's peaks past 1.0; this bounds
    them by ~1 dB of gain reduction on the loudest 1% of samples instead of
    clipping them flat.
    """
    mag = np.abs(a)
    over = mag > thr
    if not over.any():
        return a
    y = a.copy()
    y[over] = np.sign(a[over]) * (thr + (ceil - thr)
                                  * np.tanh((mag[over] - thr) / (ceil - thr)))
    return y


def build_bed(target_sec):
    """Loop the instrumental fragments (or the old bgm.wav) into a bed.

    Unchanged from v5 -- the bed material and level were accepted as-is.
    """
    srcs = sorted(glob.glob(os.path.join(ROOT, "outputs", "tts", "*.wav")))
    parts = []
    for p in srcs:
        nm = os.path.basename(p)
        if not (nm.startswith("bg_") or nm.startswith("rate_")):
            continue
        for g in make_bgm.music_parts(p):
            if len(g) > SR // 4:
                parts.append(g)
    if parts:
        bed = parts[0]
        for p in parts[1:]:
            bed = make_bgm.crossfade(bed, p)
        print(f"  bed source: {len(parts)} instrumental fragments")
    else:
        cands = [os.path.join(PUBLIC_AUDIO, "bgm.wav"),
                 os.path.join(ROOT, "outputs", "old_audio_backup_20261014", "bgm.wav")]
        old = next((c for c in cands if os.path.exists(c)), None)
        if not old:
            raise RuntimeError("no BGM source (fragments or old bgm.wav)")
        bed = read_wav_f(old)
        bed = bed[int(2.2 * SR): max(int(2.2 * SR) + 1, len(bed) - int(3.2 * SR))]
        print(f"  bed source: old bgm.wav, fades stripped ({len(bed) / SR:.2f}s)")
    fade = min(SR, len(bed) // 4)
    head, tail = bed[:fade], bed[-fade:]
    seam = head * np.linspace(1, 0, fade)[:, None] + tail * np.linspace(0, 1, fade)[:, None]
    loop = np.concatenate([bed[fade:-fade], seam])
    reps = int(np.ceil(target_sec / (len(loop) / SR))) + 1
    full = np.concatenate([loop] * reps)[: int(target_sec * SR)]
    fi, fo = int(2.0 * SR), int(3.0 * SR)
    full[:fi] *= np.linspace(0, 1, fi)[:, None]
    full[-fo:] *= np.linspace(1, 0, fo)[:, None]
    print(f"  bed: {len(full) / SR:.2f}s")
    return full


def duck_curve(narr, speech_db):
    """Per-block gain: ducked under speech, 1.0 in the gaps."""
    db = block_db(narr)
    active = db > (speech_db - 15.0)
    n = len(active)
    tgt_gain = 10 ** (-DUCK_DB / 20.0)
    att = 1 - math.exp(-0.01 / 0.02)
    rel = 1 - math.exp(-0.01 / 0.40)
    g = np.empty(n, np.float32)
    cur = 1.0
    for i in range(n):
        tgt = tgt_gain if active[i] else 1.0
        a = att if tgt < cur else rel
        cur += (tgt - cur) * a
        g[i] = cur
    g_full = np.repeat(g, WIN)
    if len(g_full) < len(narr):
        g_full = np.concatenate([g_full, np.ones(len(narr) - len(g_full), np.float32)])
    return g_full[: len(narr)], active


def report(narr, ducked, scenes, speech_db, active, g_full):
    """Level report + independently measured scene-onset error.

    The onset check is the honest one: it reads the narration envelope, not the
    ASR-derived timings, so it says whether the picture really does start where
    the voice does.
    """
    total = len(narr) / SR
    db = block_db(narr)
    voiced = db > (speech_db + PAUSE_GATE_DB)
    sil = 1.0 - voiced.mean()
    print(f"[timing] total {total:.2f}s | silence {sil * 100:.1f}%", end=" ")

    def first_voiced(t0, t1):
        i0 = max(0, int(t0 * SR) // WIN)
        i1 = min(len(voiced), int(t1 * SR) // WIN)
        w = np.where(voiced[i0:i1])[0]
        return (i0 + w[0]) * WIN / SR if len(w) else None

    errs = []
    checked = 0
    for s in scenes:
        t0 = s["startFrames"] / FPS
        want = t0 + (s["lines"][0]["start"] if s["lines"] else LEAD)
        # only a scene whose first line starts after real silence can be onset-
        # checked: consecutive lines run into each other, and then the first
        # voiced block in the window is the previous line's tail, not this one.
        if voiced[max(0, int((want - 0.30) * SR) // WIN): max(1, int((want - 0.05) * SR) // WIN)].any():
            continue
        got = first_voiced(want - 0.30, want + 1.0)
        if got is not None:
            errs.append(got - want)
            checked += 1
    if errs:
        e = np.array(errs)
        print(f"| onset err (from silence, {checked}/{len(scenes)} scenes) "
              f"median {np.median(e):+.2f}s max {np.abs(e).max():.2f}s")
    else:
        print(f"| onset not measurable (speech is continuous)")
    print()
    sp = []
    for s in scenes:
        base = s["startFrames"] / FPS
        for ln in s["lines"]:
            sp.append((base + ln["start"], base + ln["start"] + ln["dur"]))
    sp.sort()
    gaps, prev = [], 0.0
    for a, b in sp:
        if a - prev > 0.35:
            gaps.append((prev + 0.15, a - 0.15))
        prev = max(prev, b)
    if total - prev > 0.35:
        gaps.append((prev + 0.15, total))

    def avg_db(arr, wins):
        vals = []
        for a, b in wins:
            i, j = int(a * SR), int(b * SR)
            if j - i > SR // 10:
                vals.append(rms_db(arr[i:j]))
        if not vals:
            return float("nan")
        p = sum(10 ** (v / 10.0) for v in vals) / len(vals)
        return 10 * math.log10(max(p, 1e-18))

    print(f"[levels] narration speech {avg_db(narr, sp):6.1f} dB | "
          f"BGM in gaps {avg_db(ducked, gaps):6.1f} dB | "
          f"BGM under speech {avg_db(ducked, sp):6.1f} dB | "
          f"duck {avg_db(ducked, gaps) - avg_db(ducked, sp):4.1f} dB")

    a = np.asarray(active, bool)
    nb = len(a)

    def runs(val):
        out, i = [], 0
        while i < nb:
            if bool(a[i]) == val:
                j = i
                while j < nb and bool(a[j]) == val:
                    j += 1
                out.append((i, j))
                i = j
            else:
                i += 1
        return out

    def blk_db(sig, mask):
        b = sig.mean(1)[: nb * WIN].reshape(nb, WIN)[mask]
        if b.size == 0:
            return float("nan")
        return 20 * math.log10(max(float(np.sqrt((b ** 2).mean())), 1e-9))

    m_s = np.zeros(nb, bool)
    for i0, j0 in runs(True):
        if j0 - i0 > 40:
            m_s[i0 + 20: j0 - 20] = True
    silent = [(i0, j0) for i0, j0 in runs(False) if (j0 - i0) * WIN / SR > 0.5]
    m_g = np.zeros(nb, bool)
    for i0, j0 in silent:
        m_g[i0 + 45: j0] = True
    print(f"  duck (gate): narration {blk_db(narr, m_s):6.1f} dB | BGM under "
          f"speech {blk_db(ducked, m_s):6.1f} dB | BGM in gaps "
          f"{blk_db(ducked, m_g):6.1f} dB | "
          f"{blk_db(ducked, m_s) - blk_db(ducked, m_g):4.1f} dB over "
          f"{sum(j0 - i0 for i0, j0 in silent) * WIN / SR:.1f}s silence")


# ----------------------------------------------------------------------- main
def main():
    os.makedirs(V6, exist_ok=True)
    os.makedirs(PUBLIC_AUDIO, exist_ok=True)
    force = "--force" in sys.argv
    key = load_key()
    script = json.load(open(SCRIPT, encoding="utf-8"))
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    if os.path.exists(TIMINGS) and not os.path.exists(os.path.join(V6, "timings_v5_backup.json")):
        shutil.copy(TIMINGS, os.path.join(V6, "timings_v5_backup.json"))

    groups = [script[:5], script[5:]]
    texts = {f"part{i+1}": "".join("".join(s["lines"]) for s in g)
             for i, g in enumerate(groups)}
    print(f"RATE={RATE} TRIM_MIN={TRIM_MIN} TRIM_TO={TRIM_TO} PART_GAP={PART_GAP}")

    # ---- 1. rolls, chosen for pitch consistency
    print("=== TTS rolls (rate %.2f, voice %s) ===" % (RATE, VOICE))
    p1, u1, f1, d1 = pick_roll(texts["part1"], "part1", key, cache, target=None)
    p2, u2, f2, d2 = pick_roll(texts["part2"], "part2", key, cache, target=f1)

    # ---- 2. one pitch for the whole narration, small corrections only
    print("=== pitch alignment ===")
    aligned = align_pitch(p1, p2, f1, f2)

    # ---- 3. ASR word timestamps -> content alignment -> line times
    print("=== ASR word timestamps + content alignment ===")
    words, lines_by_part, cutmap, trimmed = {}, {}, {}, {}
    for name in ("part1", "part2"):
        pc = cache["parts"][name]
        if "words" not in pc:
            w, heard = asr_words(pc["url"], key)
            pc["words"] = w
            pc["heard"] = heard
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {name}: ASR {len(w)} words, heard {len(heard)} chars")
        words[name] = pc["words"]
        heard = pc.get("heard", "")
        print(f"  {name}: {len(words[name])} word timestamps from cache")

        # ---- 4. pause trim (after the pitch shift: the shift does not change
        #         the duration, and cutting the aligned file keeps the seams clean)
        a = read_wav_f(aligned[name])
        sdb = speech_rms_db(a)
        cuts, _v = trim_plan(a, sdb)
        kept = (len(a) - sum(e - s for s, e in cuts)) / SR
        print(f"  {name}: {len(a)/SR:.2f}s -> {kept:.2f}s "
              f"({len(cuts)} pauses trimmed, saved "
              f"{sum(e - s for s, e in cuts)/SR:.2f}s)")
        trimmed[name] = apply_trim(a, cuts)
        cutmap[name] = cut_map(cuts)

    # ---- 5. line times, then the scene layout (audio is the authority)
    print("=== scene layout ===")
    prot = {"part1": 0.0}
    scenes = []
    chain_end = 0.0      # absolute end of the last line of the previous part
    for gi, g in enumerate(groups):
        name = f"part{gi + 1}"
        lines_all = [ln for sc in g for ln in sc["lines"]]
        tsec, match, na, nb = line_times(lines_all, words[name])
        heard = cache["parts"][name].get("heard", "")
        print(f"  {name}: content match {match:.0f}% "
              f"(script {na} chars vs ASR {nb}), heard: {heard[:80]}")
        if match < 60:
            print("  !! low match, the timings below are unreliable")
        cm = cutmap[name]
        fixed = []
        for i, t in enumerate(tsec):
            if t is None:
                fixed.append(None)
                continue
            s, e = cm(t[0]), cm(t[1])
            fixed.append((s, e))
        # fill any unmapped line from its neighbours
        for i, t in enumerate(fixed):
            if t is None:
                prev = next((fixed[j] for j in range(i - 1, -1, -1) if fixed[j]), None)
                nxt = next((fixed[j] for j in range(i + 1, len(fixed)) if fixed[j]), None)
                a0 = prev[1] if prev else 0.0
                a1 = nxt[0] if nxt else a0 + 1.0
                fixed[i] = (a0, max(a1, a0 + 0.3))
        # ASR char interpolation jitters a boundary by up to ~0.3s, which showed
        # up as line k ending after line k+1 began. Split any overlap at its
        # midpoint so the line chain is strictly monotone.
        for i in range(1, len(fixed)):
            s0, e0 = fixed[i - 1]
            s1, e1 = fixed[i]
            if s1 < e0 - 0.12:
                mid = (s1 + e0) / 2
                fixed[i - 1] = (s0, mid)
                fixed[i] = (mid, e1)
            elif s1 < e0:
                fixed[i] = (e0, e1)

        origin = 0.0 if gi == 0 else chain_end + TAIL + PART_GAP
        prot[name] = origin
        li, prev_end = 0, 0.0
        for sc in g:
            n = len(sc["lines"])
            rel = []
            for k in range(n):
                s0, e0 = fixed[li + k]
                if k + 1 < n:
                    e0 = min(e0, max(fixed[li + k + 1][0], s0 + 0.25))
                if k > 0 and s0 < prev_end + 0.02:
                    s0 = prev_end + 0.02
                rel.append({"text": sc["lines"][k], "_s": s0,
                            "_e": max(e0, s0 + 0.25)})
                prev_end = rel[-1]["_e"]
            base = rel[0]["_s"]
            start_frames = int(round(max(origin + base - LEAD, 0.0) * FPS))
            for r in rel:
                r["start"] = round(r["_s"] - base + LEAD, 3)
                r["dur"] = round(r["_e"] - r["_s"], 3)
                r.pop("_s"), r.pop("_e")
            scenes.append({
                "id": sc["id"], "name": sc["name"],
                "audio": "audio/final_mix.wav",
                "audioOffset": round(start_frames / FPS, 3),
                "title": sc["title"], "sub": sc["sub"],
                "kind": sc["kind"], "visual": sc["visual"],
                "startFrames": start_frames, "lines": rel,
                "dur": 0.0, "durFrames": 0,
            })
            li += n
        chain_end = origin + prev_end

    # scene durations tile the timeline exactly; the last one ends after its
    # last line. Never push a scene past the next one's start -- overlapping
    # Sequences in Remotion would cut the previous scene's last subtitle short.
    last = scenes[-1]
    last_end = (last["startFrames"] / FPS + last["lines"][-1]["start"]
                + last["lines"][-1]["dur"])
    for i, s in enumerate(scenes):
        if i + 1 < len(scenes):
            end_frames = scenes[i + 1]["startFrames"]
        else:
            end_frames = int(round(max(last_end + TAIL,
                                       s["startFrames"] / FPS + 1.0) * FPS))
        s["durFrames"] = end_frames - s["startFrames"]
        s["dur"] = round(s["durFrames"] / FPS, 3)
    total_frames = scenes[-1]["startFrames"] + scenes[-1]["durFrames"]
    total_sec = total_frames / FPS
    for s in scenes:
        print(f"    {s['id']} start={s['startFrames'] / FPS:7.2f}s "
              f"dur={s['dur']:6.2f}s ({s['durFrames']}f) {len(s['lines'])} lines")

    # ---- 6. loudness match + narration timeline
    # The API's takes come back near full scale (part2_r4 has 192 clipped samples
    # in the raw file, spread over its whole length), so matching the halves by
    # RMS would drive part2's peaks to 1.38. Soft knee instead: transparent below
    # 0.75, asymptotic to 0.95, which also rounds off the provider's own clipped
    # peaks rather than hard-limiting them flat.
    levels = {n: loud_db(trimmed[n]) for n in ("part1", "part2")}
    norm, achieved, lim_frac = {}, {}, {}

    def set_level(n, ref_db):
        """Gain a take to ref_db, closing the loop on the measured level: the
        soft knee is nonlinear, so a straight gain-to-target lands off by up to
        a dB. Feedback converges in two or three passes."""
        gain_db = ref_db - levels[n]
        for _ in range(5):
            loud = trimmed[n] * 10 ** (gain_db / 20.0)
            norm[n] = soft_limit(loud)
            achieved[n] = loud_db(norm[n])
            err = ref_db - achieved[n]
            if abs(err) < 0.05:
                break
            gain_db += err
        lim_frac[n] = float((np.abs(loud) > 0.75).mean())
        cut = 20 * math.log10(max(float(np.abs(norm[n]).max()), 1e-9)
                              / max(float(np.abs(loud).max()), 1e-9))
        print(f"  {n}: loud {levels[n]:.1f} dB {gain_db:+.1f} dB"
              f" -> {achieved[n]:.1f} dB | knee {lim_frac[n] * 100:.2f}% of "
              f"samples, peak {float(np.abs(norm[n]).max()):.3f}"
              + (f", knee cuts peak {cut:.2f} dB" if cut < -0.01 else "")
              + f" | old ruler {speech_rms_db(norm[n]):.1f} dB")

    ref = sum(levels.values()) / 2
    for n in ("part1", "part2"):
        set_level(n, ref)
    for n in ("part1", "part2"):
        write_wav_f(os.path.join(V6, n + "_norm.wav"), norm[n])
    print(f"  take match: target {ref:.1f} dB, part1 {achieved['part1']:.2f} / "
          f"part2 {achieved['part2']:.2f}, within "
          f"{abs(achieved['part1'] - achieved['part2']):.2f} dB")

    buf_len = int((total_sec + 0.25) * SR)
    narr = np.zeros((buf_len, 2), np.float32)
    for gi, g in enumerate(groups):
        name = f"part{gi + 1}"
        a = norm[name]
        off = int(round(prot[name] * SR))
        f = int(0.03 * SR)
        seg = a.copy()
        if len(seg) > 2 * f:
            seg[:f] *= np.linspace(0, 1, f)[:, None]
            seg[-f:] *= np.linspace(1, 0, f)[:, None]
        end = min(buf_len, off + len(seg))
        narr[off:end] = seg[: end - off]
    write_wav_f(os.path.join(V6, "narration_full.wav"), narr)

    # Mix reference stays on the older dB-mean ruler, which the bed level and
    # duck gate were tuned against, and is measured on the placed narration
    # itself rather than on the takes.
    target = speech_rms_db(narr)
    print(f"  narration reference {target:.1f} dB (bed at "
          f"{target + BGM_GAP_DB:.1f} dB)")

    # ---- 7. BGM bed, ducking, mix
    bed = build_bed(total_sec + 0.5)
    bed = bed[:buf_len] if len(bed) >= buf_len else np.concatenate(
        [bed, np.zeros((buf_len - len(bed), 2), np.float32)])
    bed_db = rms_db(bed)
    bed *= 10 ** ((target + BGM_GAP_DB - bed_db) / 20.0)
    print(f"  bed level: {bed_db:.1f} dB -> {target + BGM_GAP_DB:.1f} dB "
          f"(gaps, speech {target:.1f} dB)")
    g_full, active = duck_curve(narr, target)
    ducked = bed * g_full[:, None]
    mix = narr + ducked
    peak = float(np.abs(mix).max())
    if peak > 0.97:
        print(f"  limiting mix: peak {peak:.3f} -> 0.97 "
              f"({20 * math.log10(0.97 / peak):+.2f} dB on the whole mix)")
        mix *= 0.97 / peak
    flat = int((np.abs(mix) >= 0.999).sum())
    ov = [f"{a['id']}->{b['id']}" for a, b in zip(scenes, scenes[1:])
          if a["startFrames"] + a["durFrames"] > b["startFrames"]]
    print(f"  clipped samples {flat} | scene overlaps {ov or 'none'} | "
          f"peak {float(np.abs(mix).max()):.3f}")
    fo = int(1.2 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
    write_wav_f(os.path.join(V6, "ducked_bgm.wav"), ducked)
    write_wav_f(os.path.join(V6, "final_mix.wav"), mix)
    final = os.path.join(PUBLIC_AUDIO, "final_mix.wav")
    write_wav_f(final, mix)
    print(f"\n  final_mix.wav: {len(mix) / SR:.2f}s, "
          f"{os.path.getsize(final) // (1024 * 1024)} MB")

    # ---- 8. report + outputs
    report(narr, ducked, scenes, target, active, g_full)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp3 = os.path.join(V6, "final_mix.mp3")
    subprocess.run([exe, "-y", "-i", final, "-c:a", "libmp3lame",
                    "-b:a", "192k", mp3], capture_output=True)
    print("  mp3 preview:", mp3)
    json.dump({"fps": FPS, "durationInFrames": total_frames, "scenes": scenes},
              open(TIMINGS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  timings.json: {total_frames} frames = {total_sec:.2f}s")
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print("\ndone. re-render next: npx remotion render src/index.ts "
          "AdventureIsland out/adventure_island_story.mp4")


if __name__ == "__main__":
    main()
