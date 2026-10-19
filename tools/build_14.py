"""build_14.py — narration build for the 1-2 → 1-4 video (video14).

This is build_v6.py's audio chain, generalised from 2 parts to N parts and
pointed at the new project. Every rule below was measured on the 1-1 video and
is recorded in outputs/v6/ and the 解说音频问题复盘:

  1. RATE 0.7. 0.5 (211 字/分) drags the vowels and reads as slow playback;
     0.6 = 255, 0.7 = 282, 1.0 = 426. One voice + one style for every part.
  2. Pitch consistency by RE-ROLLING, not by shifting. v5 forced part2 down
     -4.18 st and lost 20% of the detected syllable onsets plus 0.5 pt of the
     2-8 kHz energy. Here a take is re-rolled until it lands near part1 and
     only the residual (<= MAX_RESID_ST) is corrected, formants preserved.
  3. Pauses longer than TRIM_MIN are cut back to TRIM_TO. v5 left 24% of the
     runtime silent (15 pauses > 0.4 s, 4.5 s between parts).
  4. Line timings from ASR WORD timestamps + a difflib content alignment over
     a normalised char stream. v5 divided sentence timestamps by char count,
     which is exactly why every scene ran 1.7-7.4 s late.
  5. Loudness matched in the energy domain (loud_db). Averaging dB values gave
     a fake 4.2 dB gap and left a real 2.9 dB step after "normalising".
  6. BGM is the game's own music, ducked under the voice (side-chain style).

Usage:  python tools/build_14.py [--force]
        RATE / TRIM_MIN / TRIM_TO / PART_GAP / MAX_PART_CHARS env overrides.
        Rolls and ASR results are cached in outputs/v14/v14_cache.json, so the
        timing can be re-tuned without new API calls.
"""
import bisect
import difflib
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import wave

import numpy as np
import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_batch import END, MODEL, load_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video14", "script.json")
PUBLIC_AUDIO = os.path.join(ROOT, "video14", "public", "audio")
TIMINGS = os.path.join(ROOT, "video14", "src", "timings.json")
V14 = os.path.join(ROOT, "outputs", "v14")
ROLLS = os.path.join(V14, "rolls")
CACHE = os.path.join(V14, "v14_cache.json")

VOICE = os.environ.get("VOICE", "longsanshu_v3.1")
STYLE = (
    "成熟的中文男声纪录片旁白，声音磁性低沉有质感，情绪饱满富有感染力，"
    "语气笃定从容、张弛有度，像讲述一段亲身经历的技术故事，节奏自然，发音清晰。"
    "全篇保持同一个低沉稳定的音色和音高，不要因为情绪起伏抬高音调，"
    "整体像一个人在同一个录音棚里一口气讲完。"
)

RATE = float(os.environ.get("RATE", "0.7"))

F0_TARGET = 118.0
F0_ACCEPT = (110.0, 126.0)
# Wider than v6's (105, 125): this script needs 8 takes, not 2, and some
# passages come back a few semitones high no matter how often they are rolled
# (part3 landed 149-173 Hz over ten rolls). Clamping the shared target down to
# 125 would force every one of those outliers to the end of the range and past
# MAX_RESID_ST, so the target is allowed to sit where the takes actually are.
F0_CLAMP = (110.0, 138.0)
# v6 allowed 1.5 st. Eight takes span more ground than two, so 2.5 st is the
# ceiling here -- still well short of the 4.18 st that cost v5 20% of its
# detected syllable onsets.
MAX_RESID_ST = float(os.environ.get("MAX_RESID_ST", "2.5"))
# DO NOT shrink this: corrections below it are inaudible and every one of them
# costs a pass of rubberband, which is what makes the voice sound processed.
SKIP_ST = float(os.environ.get("SKIP_ST", "0.5"))
# v6 measured the same text coming back anywhere from 113 to 160 Hz -- roughly
# six semitones of spread -- so ten rolls, not six, is what it actually takes to
# land next to part1 without a big correction. Finished parts are cached, so a
# resumed run only pays for the ones that are still missing.
ROLLS_MAX = int(os.environ.get("ROLLS_MAX", "10"))

# A seed is a second, independent way to get a different rendition -- and the
# only one that leaves rate, voice and style untouched. Some passages are stuck
# in a high register: part3's text came back 149-173 Hz over ten unseeded rolls,
# but seed 1 on that same text returned 120.4 Hz. So a roll that misses the
# window tries the next seed instead of giving up and leaning on a pitch shift.
SEEDS = [1, 3, 7, 11, 5, 13, 17, 23, 29]

SR = 48000
FPS = 30
LEAD = 0.15          # scene starts this much before its first line
TAIL = 0.5           # last scene keeps this much after its last line
# Parts are only a synthesis-boundary artefact (the model caps one request at
# ~60 s), not a content boundary, so the seam is a breath, not a scene gap.
PART_GAP = float(os.environ.get("PART_GAP", "0.25"))
# v6's bed sat -8 dB under the narration reference, which measured as ~24 dB
# below the voice once ducked. Here the scenes tile the timeline with no gaps
# (the only BGM-up moments are the 7 part seams, ~1.6% of the runtime), so at
# -9 dB the game music was effectively inaudible for the whole video. -5 dB puts
# it about 20 dB under the voice: present, still clearly the backing track.
BGM_GAP_DB = float(os.environ.get("BGM_GAP_DB", "-5.0"))
DUCK_DB = float(os.environ.get("DUCK_DB", "12.0"))
WIN = 480            # 10 ms envelope block

# Tighter than v6's 0.50 / 0.34 / 0.15 / 0.40. v6 had 2 parts and 462 chars over
# 87 s; this script has 8 parts and 120 short lines over ~276 s, so both things
# that v6 trimmed once now happen eight times as often: every part carries its
# own head + tail, and every line ends in a "。" the model pauses on. At v6's
# settings the first build came out at 19.5% silence (v5 was 24%, v6 9%).
TRIM_MIN = float(os.environ.get("TRIM_MIN", "0.35"))
TRIM_TO = float(os.environ.get("TRIM_TO", "0.24"))
HEAD_TO = float(os.environ.get("HEAD_TO", "0.10"))
TAIL_TO = 0.30
PAUSE_GATE_DB = -22.0

# The model caps one request at ~60 s of audio; 282 字/分 => ~4.7 字/s, so stay
# well under that. Scenes are never split across parts.
MAX_PART_CHARS = int(os.environ.get("MAX_PART_CHARS", "240"))

# BGM: the game's own music. Ranges are chosen to avoid the death/clear jingles
# so the bed stays neutral under the voice.
BED_SOURCES = [("game14.wav", 4.0, 70.0), ("game12.wav", 3.0, 45.0)]

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
    how loud it is -- a take with more low-level material reads several dB
    quieter than it sounds, and matching on it left a 2.9 dB step between the
    two takes in v6. Weighting each block by its energy is what tracks
    loudness.
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
    mono = a.mean(1)
    n = len(mono) // WIN
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(e, 1e-9))


def measure_f0(path):
    """Median F0 over voiced frames (Praat). The ruler v5 established: it reads
    a synthetic 150 Hz tone as 150.0 and a known -3 st rubberband shift as
    -2.97."""
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
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
                    "-af", f"rubberband=pitch={ratio:.6f}:formant=preserved",
                    "-c:a", "pcm_s16le", dst], check=True)


# ------------------------------------------------------------------ TTS rolls
def synth_roll(text, key, tag, seed=None):
    dst = os.path.join(ROLLS, tag + ".wav")
    params = {"voice": VOICE, "instruction": STYLE, "rate": RATE}
    if seed is not None:
        params["seed"] = seed
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": text},
        "parameters": params,
    }, ensure_ascii=False).encode("utf-8")
    # 403s come back in bursts (rate control, not auth): back off hard instead
    # of hammering, and keep the cache warm so a resumed run skips finished parts.
    for attempt in range(8):
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
            print(f"    retry {attempt + 1} {tag}: {e}", flush=True)
            time.sleep(8 + attempt * 6)
    raise RuntimeError("synth failed: " + tag)


def sha_of(text, target):
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    # Deliberately excludes SEEDS: a cached take that already landed in the
    # window is still the take to use, and adding the seed list to the hash
    # would throw away every finished part the moment the sweep changes.
    h.update(f"|{VOICE}|{RATE}|{STYLE}|{target}".encode("utf-8"))
    return h.hexdigest()[:16]


def pick_roll(text, name, key, cache, target=None):
    """Roll the part until its pitch is usable; return (wav, url, f0, dur).

    With no target (part1) the take is accepted as soon as it lands in the deep
    register we want; otherwise it is scored by distance to part1's pitch and
    the closest take wins, so the residual correction stays small.
    """
    want = sha_of(text, target)
    pc0 = cache.get("parts", {}).get(name, {})
    if pc0.get("sha") == want and os.path.exists(pc0.get("wav", "")):
        print(f"  {name}: cached roll {os.path.basename(pc0['wav'])} "
              f"F0 {pc0['f0']:.1f} Hz {pc0['dur']:.2f}s")
        return pc0["wav"], pc0["url"], pc0["f0"], pc0["dur"]

    best = None
    for i in range(ROLLS_MAX):
        seed = None if i == 0 else SEEDS[(i - 1) % len(SEEDS)]
        tag = f"{name}_r{i}" + ("" if seed is None else f"s{seed}")
        wav, url, dur = synth_roll(text, key, tag, seed)
        f0 = measure_f0(wav)
        cpm = len(text) / dur * 60
        st = None if target is None else 12 * math.log2(f0 / target)
        print(f"    roll {tag}: {dur:6.2f}s  {cpm:5.0f} 字/分  F0 {f0:5.1f} Hz"
              + ("" if st is None else f"  ({st:+.2f} st vs part1)"), flush=True)
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
    wav, url, f0, _score, dur = best
    st = None if target is None else 12 * math.log2(f0 / target)
    if st is not None and abs(st) > MAX_RESID_ST:
        print(f"    !! best take is still {st:+.2f} st off; correction will be "
              f"clamped to {MAX_RESID_ST} st")
    print(f"    -> {name} uses {os.path.basename(wav)} (F0 {f0:.1f} Hz)")
    cache.setdefault("parts", {})[name] = {
        "sha": want, "wav": wav, "url": url, "f0": f0, "dur": dur,
        "rate": RATE, "voice": VOICE}
    return wav, url, f0, dur


def align_pitch(parts, f0s):
    """Bring every part to one pitch, with only a small correction allowed.

    The target is the MEDIAN of the takes, not the mean. Two of the eight
    passages come back around 145 Hz however they are rolled; the mean drags
    to 129 Hz and every single part then needs a 0.5-2.2 st nudge to reach it,
    which is worse than leaving them alone - the point of the whole exercise
    is that rubberband is what makes the voice sound like a different, sleepy
    narrator, so the win is in how many parts never touch it at all. The
    median ignores the outliers: with it, six of the eight land on target as
    generated, and only the two genuinely high passages get corrected.

    Parts already within SKIP_ST are left completely untouched. Half a
    semitone is less than the pitch variation of one person speaking, so
    "correcting" it only buys artefacts.
    """
    f1 = f0s["part1"]
    vals = sorted(v for n in parts if f0s[n] for v in [f0s[n]])
    mid = len(vals) // 2
    med = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    mean = sum(vals) / len(vals)
    target = min(max(med, F0_CLAMP[0]), F0_CLAMP[1])
    print(f"  part1 {f1:.1f} Hz | median {med:.1f} Hz (mean {mean:.1f} Hz, "
          f"spread {vals[0]:.0f}-{vals[-1]:.0f}) -> target {target:.1f} Hz")
    out = {}
    for name in parts:
        src = parts[name]
        f0 = f0s[name]
        dst = os.path.join(ROLLS, f"{os.path.basename(src)[:-4]}_aligned.wav")
        semis = 12 * math.log2(target / f0) if f0 else 0.0
        if abs(semis) < SKIP_ST:
            print(f"  {name}: F0 {f0:.1f} Hz already on target, left untouched")
            out[name] = src
            continue
        if abs(semis) > MAX_RESID_ST:
            print(f"  {name}: correction {semis:+.2f} st clamped to "
                  f"{math.copysign(MAX_RESID_ST, semis):+.2f} st")
            semis = math.copysign(MAX_RESID_ST, semis)
        ratio = 2 ** (semis / 12)
        got = f0
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
    gaps = [12 * math.log2(measure_f0(out[n]) / measure_f0(out["part1"]))
            for n in parts if n != "part1"]
    print(f"  aligned gap part1->others: "
          + ", ".join(f"{g:+.2f} st" for g in gaps))
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

    Numerals and latin runs collapse to '#': the ASR spells numbers and names
    differently from the script ("66号" vs "六十六号", "boss" vs "B-O-S-S"), and
    on raw char counts that mismatch is what accumulated into the scene drift.
    """
    if c.isdigit() or c in "一二三四五六七八九十百千万零两":
        return "#"
    c = c.lower()
    if c.isascii() and c.isalnum():
        return "#"
    return c if c.isalnum() else ""


def char_times(words):
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
    """Content-align the script lines onto ASR char times."""
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
        out.append((times(idxs[0])[0], times(idxs[-1])[1]))
    return out, 100.0 * len(m) / max(1, len(A)), len(A), len(B)


# --------------------------------------------------------------- pause trim
def trim_plan(a, speech_db):
    db = block_db(a)
    voiced = db > (speech_db + PAUSE_GATE_DB)
    nb = len(voiced)
    runs, i = [], 0
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
        drop = int((e - s) - keep * SR)
        mid = (s + e) // 2
        cuts.append((mid - drop // 2, mid - drop // 2 + drop))
    return cuts, voiced


def cut_map(cuts):
    def f(t):
        s = 0.0
        for a, b in cuts:
            if t * SR <= a:
                break
            s += (min(t * SR, b) - a) / SR
        return t - s
    return f


def apply_trim(a, cuts):
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
        seam = (acc[-m:] * np.linspace(1, 0, m)[:, None]
                + sg[:m] * np.linspace(0, 1, m)[:, None])
        acc = np.concatenate([acc[:-m], seam, sg[m:]])
    return acc


# ------------------------------------------------------------------ BGM / mix
def soft_limit(a, thr=0.75, ceil=0.95):
    """Soft knee: transparent below thr, asymptotic to ceil, C1 continuous.

    The provider's takes arrive near full scale, so matching parts by RMS would
    drive one part's peaks past 1.0; this bounds them by ~1 dB of gain
    reduction on the loudest samples instead of clipping them flat.
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
    """Loop the game's own music into a bed.

    The 1-1 video's bed was stitched out of instrumental TTS fragments because
    the game audio was not available; here the real thing exists, so the bed is
    a crossfaded loop of the two playthrough tracks, trimmed to skip the death
    and stage-clear jingles.
    """
    segs = []
    for name, t0, t1 in BED_SOURCES:
        p = os.path.join(PUBLIC_AUDIO, name)
        if not os.path.exists(p):
            print(f"  !! bed source missing: {p}")
            continue
        a = read_wav_f(p)
        seg = a[int(t0 * SR): min(len(a), int(t1 * SR))]
        if len(seg) > SR:
            segs.append(seg)
    if not segs:
        raise RuntimeError("no BGM source")
    bed = segs[0]
    for s in segs[1:]:
        n = min(SR // 2, len(bed), len(s))
        f = np.linspace(0, 1, n)[:, None]
        bed = np.concatenate([bed[:-n], bed[-n:] * (1 - f) + s[:n] * f, s[n:]])
    print(f"  bed raw: {len(bed) / SR:.2f}s from {len(segs)} game tracks")

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


def report(narr, ducked, scenes, speech_db, active):
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

    errs, checked = [], 0
    for s in scenes:
        t0 = s["startFrames"] / FPS
        want = t0 + (s["lines"][0]["start"] if s["lines"] else LEAD)
        lo = max(0, int((want - 0.30) * SR) // WIN)
        hi = max(1, int((want - 0.05) * SR) // WIN)
        if voiced[lo:hi].any():
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
        print("| onset not measurable (speech is continuous)")
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
def pack_parts(script):
    """Greedy: fill each part up to MAX_PART_CHARS, never splitting a scene."""
    groups, cur, n = [], [], 0
    for sc in script:
        c = sum(len(x) for x in sc["lines"])
        if cur and n + c > MAX_PART_CHARS:
            groups.append(cur)
            cur, n = [], 0
        cur.append(sc)
        n += c
    if cur:
        groups.append(cur)
    return groups


def main():
    os.makedirs(V14, exist_ok=True)
    os.makedirs(ROLLS, exist_ok=True)
    os.makedirs(PUBLIC_AUDIO, exist_ok=True)
    key = load_key()
    script = json.load(open(SCRIPT, encoding="utf-8"))
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

    groups = pack_parts(script)
    names = [f"part{i + 1}" for i in range(len(groups))]
    texts = {nm: "".join("".join(sc["lines"]) for sc in g)
             for nm, g in zip(names, groups)}
    print(f"RATE={RATE} TRIM_MIN={TRIM_MIN} TRIM_TO={TRIM_TO} "
          f"PART_GAP={PART_GAP} MAX_PART_CHARS={MAX_PART_CHARS}")
    print(f"{len(groups)} parts: "
          + ", ".join(f"{nm}={len(t)}字" for nm, t in texts.items()))

    # ---- 1. rolls, chosen for pitch consistency
    print("=== TTS rolls (rate %.2f, voice %s) ===" % (RATE, VOICE))
    raw, f0s, durs = {}, {}, {}
    for i, nm in enumerate(names):
        tgt = None if i == 0 else f0s["part1"]
        raw[nm], _u, f0s[nm], durs[nm] = pick_roll(texts[nm], nm, key, cache,
                                                   target=tgt)
        json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)

    # ---- 2. one pitch for the whole narration, small corrections only
    print("=== pitch alignment ===")
    aligned = align_pitch(raw, f0s)

    # ---- 3. ASR word timestamps -> content alignment -> line times
    print("=== ASR word timestamps + content alignment ===")
    words, cutmap, trimmed = {}, {}, {}
    for nm in names:
        pc = cache["parts"][nm]
        if "words" not in pc:
            w, heard = asr_words(pc["url"], key)
            pc["words"] = w
            pc["heard"] = heard
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {nm}: ASR {len(w)} words, heard {len(heard)} chars")
        words[nm] = pc["words"]
        print(f"  {nm}: {len(words[nm])} word timestamps from cache")

        # ---- 4. pause trim (after the pitch shift: the shift preserves the
        #         duration, and cutting the aligned file keeps the seams clean)
        a = read_wav_f(aligned[nm])
        sdb = speech_rms_db(a)
        cuts, _v = trim_plan(a, sdb)
        kept = (len(a) - sum(e - s for s, e in cuts)) / SR
        print(f"  {nm}: {len(a) / SR:.2f}s -> {kept:.2f}s "
              f"({len(cuts)} pauses trimmed, saved "
              f"{sum(e - s for s, e in cuts) / SR:.2f}s)")
        if len(a) / SR > 55.0:
            print(f"  !! {nm} is {len(a) / SR:.1f}s -- the model truncates around "
                  f"60 s, check the ASR match below for missing tail lines")
        trimmed[nm] = apply_trim(a, cuts)
        cutmap[nm] = cut_map(cuts)

    # ---- 5. line times, then the scene layout (audio is the authority)
    print("=== scene layout ===")
    prot = {names[0]: 0.0}
    scenes, chain_end = [], 0.0
    for gi, g in enumerate(groups):
        nm = names[gi]
        lines_all = [ln for sc in g for ln in sc["lines"]]
        tsec, match, na, nb = line_times(lines_all, words[nm])
        print(f"  {nm}: content match {match:.0f}% "
              f"(script {na} chars vs ASR {nb})")
        if match < 60:
            print("  !! low match, the timings below are unreliable")
        cm = cutmap[nm]
        fixed = [None if t is None else (cm(t[0]), cm(t[1])) for t in tsec]
        for i, t in enumerate(fixed):
            if t is None:
                prev = next((fixed[j] for j in range(i - 1, -1, -1) if fixed[j]), None)
                nxt = next((fixed[j] for j in range(i + 1, len(fixed)) if fixed[j]), None)
                a0 = prev[1] if prev else 0.0
                a1 = nxt[0] if nxt else a0 + 1.0
                fixed[i] = (a0, max(a1, a0 + 0.3))
        # ASR char interpolation jitters a boundary by up to ~0.3s, which showed
        # up as line k ending after line k+1 began. Split overlaps at the
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

        origin = 0.0 if gi == 0 else chain_end + PART_GAP
        prot[nm] = origin
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
                "id": sc["id"], "name": sc["name"], "kind": sc["kind"],
                "visual": sc["visual"], "title": sc["title"], "sub": sc["sub"],
                "clip": sc.get("clip"), "clip2": sc.get("clip2"),
                "clipStart": sc.get("clipStart", 0),
                "audio": "audio/final_mix.wav",
                "audioOffset": round(start_frames / FPS, 3),
                "startFrames": start_frames, "lines": rel,
                "dur": 0.0, "durFrames": 0,
            })
            li += n
        chain_end = origin + prev_end

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
    levels = {nm: loud_db(trimmed[nm]) for nm in names}
    norm, achieved = {}, {}

    def set_level(nm, ref_db):
        """Gain a take to ref_db, closing the loop on the measured level: the
        soft knee is nonlinear, so a straight gain-to-target lands off by up to
        a dB. Feedback converges in two or three passes."""
        gain_db = ref_db - levels[nm]
        for _ in range(5):
            loud = trimmed[nm] * 10 ** (gain_db / 20.0)
            norm[nm] = soft_limit(loud)
            achieved[nm] = loud_db(norm[nm])
            err = ref_db - achieved[nm]
            if abs(err) < 0.05:
                break
            gain_db += err
        lim = float((np.abs(loud) > 0.75).mean())
        print(f"  {nm}: loud {levels[nm]:.1f} dB {gain_db:+.1f} dB"
              f" -> {achieved[nm]:.1f} dB | knee {lim * 100:.2f}% of samples, "
              f"peak {float(np.abs(norm[nm]).max()):.3f}"
              f" | old ruler {speech_rms_db(norm[nm]):.1f} dB")

    ref = sum(levels.values()) / len(levels)
    for nm in names:
        set_level(nm, ref)
    print(f"  take match: target {ref:.1f} dB, spread "
          f"{max(achieved.values()) - min(achieved.values()):.2f} dB")
    for nm in names:
        write_wav_f(os.path.join(V14, nm + "_norm.wav"), norm[nm])

    buf_len = int((total_sec + 0.25) * SR)
    narr = np.zeros((buf_len, 2), np.float32)
    for nm in names:
        a = norm[nm]
        off = int(round(prot[nm] * SR))
        f = int(0.03 * SR)
        seg = a.copy()
        if len(seg) > 2 * f:
            seg[:f] *= np.linspace(0, 1, f)[:, None]
            seg[-f:] *= np.linspace(1, 0, f)[:, None]
        end = min(buf_len, off + len(seg))
        narr[off:end] = seg[: end - off]
    write_wav_f(os.path.join(V14, "narration_full.wav"), narr)

    # ---- 7. BGM bed, ducking, mix
    target = speech_rms_db(narr)
    print(f"  narration reference {target:.1f} dB (bed at "
          f"{target + BGM_GAP_DB:.1f} dB)")
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
    write_wav_f(os.path.join(V14, "ducked_bgm.wav"), ducked)
    write_wav_f(os.path.join(V14, "final_mix.wav"), mix)
    final = os.path.join(PUBLIC_AUDIO, "final_mix.wav")
    write_wav_f(final, mix)
    print(f"\n  final_mix.wav: {len(mix) / SR:.2f}s, "
          f"{os.path.getsize(final) // (1024 * 1024)} MB")

    # ---- 8. report + outputs
    report(narr, ducked, scenes, target, active)
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp3 = os.path.join(V14, "final_mix.mp3")
    subprocess.run([exe, "-y", "-i", final, "-c:a", "libmp3lame",
                    "-b:a", "192k", mp3], capture_output=True)
    print("  mp3 preview:", mp3)
    json.dump({"fps": FPS, "durationInFrames": total_frames,
               "scenes": scenes, "meta": {"voice": VOICE, "rate": RATE,
                                          "model": MODEL}},
              open(TIMINGS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  timings.json: {total_frames} frames = {total_sec:.2f}s")
    print("\ndone. re-render next: npx remotion render src/index.ts "
          "AdventureIsland14 video14/out/adventure_island_14.mp4")


if __name__ == "__main__":
    main()
