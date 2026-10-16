"""build_v5.py — rebuild narration + BGM for the Adventure Island video.

Driven by user requirements:
  * magnetic voice, SAME voice across the whole video (one voice id + one
    instruction for every request, loudness-matched between parts)
  * emotionally expressive delivery (style instruction, no music in the TTS)
  * NO music baked into the TTS audio. The BGM is a separate bed mixed with
    side-chain style ducking done in numpy: very quiet under speech,
    louder in the gaps.
  * replaces every old audio file in video/public/audio (moved to
    outputs/old_audio_backup_20261014/)

Pipeline:
  1. TTS   qwen-audio-3.1-tts-next, voice longsanshu_v3.1, 2 parts
           (input.text_prompt = spoken text; parameters.instruction = style;
            parameters.rate = 0.5 -- the model's default pace is 2x too fast)
  1b. pitch alignment: the model reads the two halves at different pitches with
           the same voice id (118.0 vs 152.6 Hz on the current take = 4.5
           semitones). Measured with Praat, both parts are shifted onto part1's
           pitch with ffmpeg's formant-preserving rubberband.
  2. ASR   paraformer-v2 sentence timestamps -> subtitle line timings
  3. build narration timeline (parts placed sequentially + gap) and a
     looped BGM bed, both 48 kHz stereo
  4. ducking mix in numpy -> video/public/audio/final_mix.wav
  5. write video/src/timings.json, move old audio out of public/

Usage:  python tools/build_v5.py [--force]
        VOICE env var overrides the voice id.
"""
import glob
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
import make_bgm  # noqa: E402  (reuse fragment splicing helpers)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "video", "script.json")
PUBLIC_AUDIO = os.path.join(ROOT, "video", "public", "audio")
TIMINGS = os.path.join(ROOT, "video", "src", "timings.json")
V5 = os.path.join(ROOT, "outputs", "v5")
BACKUP = os.path.join(ROOT, "outputs", "old_audio_backup_20261014")
CACHE = os.path.join(V5, "v5_cache.json")

VOICE = os.environ.get("VOICE", "longsanshu_v3.1")
# Style chosen by measurement (outputs/v5/style_test.py, style_spread.py): of
# the candidate instructions this one keeps the expressive delivery the brief
# asked for and reads both parts at a similar pace (269 / 257 字/分). None of
# them holds the pitch steady -- part2 always comes back ~4-5 semitones higher
# -- so the pitch is normalised to a common target below.
STYLE = (
    "成熟的中文男声纪录片旁白，声音磁性低沉有质感，情绪饱满富有感染力，"
    "语气笃定从容、张弛有度，像讲述一段亲身经历的技术故事，节奏自然，发音清晰。"
    "全篇保持同一个低沉稳定的音色和音高，不要因为情绪起伏抬高音调，"
    "整体像一个人在同一个录音棚里一口气讲完。"
)

# Speech rate. The model's default pace is ~7 chars/s (426 字/分), far too fast
# for narration, and it ignores pace wording in `instruction` (measured: asking
# for "slow down 30%" made it *faster*). The `rate` parameter IS honoured and
# monotonic -- measured on the same 40 chars:
#   0.5 -> 211 字/分 | 0.6 -> 255 | 0.7 -> 282 | 0.8 -> 366 | default -> 426
# -- while F0 stays at 115-120 Hz, so the voice/timbre is unchanged.
# 0.5 reproduces the original 152 s pacing this video was cut for.
RATE = float(os.environ.get("RATE", "0.5"))

# Narration pitch. The model reads the two halves at different pitches with the
# same voice id -- on the current take part1 came back at 118.0 Hz and part2 at
# 152.6 Hz, 4.5 semitones apart, which is the "sounds like two narrators" the
# brief complains about. Their formants are only ~0.8 semitones apart, so it is
# the same vocal tract reading higher, not a different voice: aligning the pitch
# with formant preservation fixes it. Every part is aligned to part1's measured
# pitch, clamped into this range so the narration stays deep.
F0_FLOOR, F0_CEIL = 105.0, 125.0

SR = 48000
FPS = 30
LEAD = 0.15        # scene starts this much before its first line
TAIL = 0.5         # scene keeps this much after its last line
PART_GAP = 1.6     # silence between the two narration parts (BGM swells here)
BGM_GAP_DB = -8.0  # BGM level in gaps, relative to the speech level
DUCK_DB = 13.0     # extra reduction under speech
WIN = 480          # 10 ms envelope block

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
    """Duration of a raw API WAV.

    These files carry a bogus data-size field in the header (a 0x1FFFFFFF style
    sentinel, so wave.getnframes() reports ~11185 s for every one of them), but
    the fmt chunk (48 kHz, 16 bit, stereo) and the real file size are correct.
    Derive the duration from the byte count instead of the header.
    """
    w = wave.open(path)
    ch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
    w.close()
    return max(0.0, (os.path.getsize(path) - 44) / float(fr * ch * sw))


def rms_db(a):
    mono = a.mean(1)
    return 20 * math.log10(max(float(np.sqrt((mono ** 2).mean())), 1e-9))


def measure_f0_numpy(path, fft=16384):
    """Fallback F0 (harmonic product spectrum) if parselmouth is unavailable.

    The stride in the harmonic multiplication matters: X[f]*X[2f]*X[3f]... needs
    X[i0*h : i1*h : h], a plain slice multiplies neighbouring bins and reads a
    synthetic 130 Hz tone as 152 Hz.
    """
    a = read_wav_f(path).mean(1)
    n = len(a) // fft
    if n < 4:
        return 0.0
    X = np.abs(np.fft.rfft(a[: n * fft].reshape(n, fft) * np.hanning(fft), axis=1))
    e = X.sum(1)
    X = X[e > e.max() * 0.15]
    if len(X) < 4:
        return 0.0
    bw = SR / fft
    i0, i1 = int(80 / bw), int(400 / bw)
    hps = X[:, i0:i1].copy()
    for h in (2, 3, 4, 5):
        hps = hps * X[:, i0 * h: i1 * h: h]
    out = []
    for i, k in enumerate(np.argmax(hps, axis=1)):
        d = 0.0
        if 0 < k < hps.shape[1] - 1:
            y0, y1, y2 = (np.log(hps[i, k - 1] + 1e-12),
                          np.log(hps[i, k] + 1e-12),
                          np.log(hps[i, k + 1] + 1e-12))
            den = y0 - 2 * y1 + y2
            if den:
                d = max(-0.5, min(0.5, 0.5 * (y0 - y2) / den))
        out.append((i0 + k + d) * bw)
    return float(np.median(out))


def measure_f0(path):
    """Median F0 of the voiced frames, measured with Praat's tracker.

    Two home-made estimators were tried first and were not good enough: on this
    voice a cepstrum and an HPS disagreed by 3 semitones, and both under-read a
    known -3 semitone shift as -2.3. Praat reads a synthetic 150 Hz tone as
    150.0, and a paired per-frame test (same instant, both files voiced) of a
    known -3 semitone rubberband shift comes back as -2.97, so this is the ruler
    the alignment is built on. The audio is read through read_wav_f because the
    raw API WAVs carry a bogus data-size field that Praat warns about.
    """
    try:
        import parselmouth
    except ImportError:
        return measure_f0_numpy(path)
    snd = parselmouth.Sound(read_wav_f(path).mean(1).astype(np.float64),
                            sampling_frequency=SR)
    v = snd.to_pitch(time_step=0.01, pitch_floor=75.0,
                     pitch_ceiling=400.0).selected_array["frequency"]
    v = v[v > 0]
    return float(np.median(v)) if len(v) else 0.0


def pitch_shift(src, dst, ratio):
    """Time-preserving pitch shift (ffmpeg rubberband, formants preserved).

    pitchq is left at its default (quality); the option takes rubberband's
    OptionPitch* constants, not a 0-3 scale.
    """
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [exe, "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-af", f"rubberband=pitch={ratio:.6f}:formant=preserved",
         "-c:a", "pcm_s16le", dst], check=True)


def align_pitch(parts):
    """Give every narration part the same pitch: part1's, clamped to F0_FLOOR..CEIL.

    The shift is verified after the fact and corrected once: the median over
    voiced frames is a statistic of a distribution, so it can move slightly less
    than the true shift when frames at the bottom of the search range lose
    voicing. The correction always re-shifts the original, never the result.
    """
    f0 = {n: measure_f0(p) for n, p in parts.items()}
    for n in sorted(parts):
        print(f"  {n}: raw F0 {f0[n]:.1f} Hz")
    target = min(max(f0["part1"], F0_FLOOR), F0_CEIL)
    print(f"  target F0 {target:.1f} Hz "
          f"(part1 {f0['part1']:.1f}, clamped to {F0_FLOOR:.0f}-{F0_CEIL:.0f})")

    out = {}
    for n, p in parts.items():
        dst = os.path.join(V5, "%s_f%d.wav" % (n, round(target)))
        if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(p):
            got = measure_f0(dst)
            print(f"  {n}: reusing {os.path.basename(dst)} (F0 {got:.1f} Hz)")
            out[n] = dst
            continue
        if not f0[n]:
            print(f"  {n}: could not measure F0, keeping the raw part")
            out[n] = p
            continue
        semis = 12 * math.log2(target / f0[n])
        if abs(semis) > 6.0:                   # keep the artefact budget sane
            semis = math.copysign(6.0, semis)
        if abs(semis) < 0.3:
            print(f"  {n}: F0 {f0[n]:.1f} Hz already at target, left untouched")
            out[n] = p
            continue
        ratio = 2 ** (semis / 12)
        for attempt in range(3):
            pitch_shift(p, dst, ratio)
            got = measure_f0(dst)
            resid = 12 * math.log2(target / got) if got else 0.0
            if abs(resid) <= 0.35:
                break
            print(f"  {n}: pass {attempt + 1} landed at {got:.1f} Hz, "
                  f"adjusting by {resid:+.2f} semitones")
            ratio *= 2 ** (resid / 12)
        d0, d1 = dur_of(p), dur_of(dst)
        print(f"  {n}: F0 {f0[n]:.1f} -> {got:.1f} Hz "
              f"({12 * math.log2(got / f0[n]):+.2f} semitones, formants "
              f"preserved, {d0:.2f}s -> {d1:.2f}s)")
        if abs(d0 - d1) > 0.05:
            print(f"  !! {n}: pitch shift changed the duration, timings are off")
        out[n] = dst
    if len(out) == 2:
        gap = 12 * math.log2(measure_f0(out["part2"]) / measure_f0(out["part1"]))
        print(f"  aligned gap part1->part2: {gap:+.2f} semitones")
    return out


def speech_rms_db(a):
    """RMS of the loud (speech) blocks only — ignores pauses."""
    mono = a.mean(1)
    n = len(mono) // WIN
    if n == 0:
        return rms_db(a)
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    loud = db[db > -45.0]
    if len(loud) == 0:
        return rms_db(a)
    return float(loud.mean())


# --------------------------------------------------------------------- TTS/ASR
def synth_part(text, name, key, style=None):
    dst = os.path.join(V5, name + ".wav")
    body = json.dumps({
        "model": MODEL,
        "input": {"text_prompt": text},          # pure narration, no BGM note
        "parameters": {"voice": VOICE, "instruction": style or STYLE, "rate": RATE},
    }, ensure_ascii=False).encode("utf-8")
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                END, data=body,
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            a = r["output"]["audio"]
            urllib.request.urlretrieve(a["url"], dst)
            return dst, a["url"], a.get("duration")
        except Exception as e:
            print(f"  retry {attempt + 1} {name}: {e}")
            time.sleep(4)
    raise RuntimeError("synth failed: " + name)


def asr_sentences(url, key):
    body = json.dumps({"model": "paraformer-v2",
                       "input": {"file_urls": [url]}}).encode()
    req = urllib.request.Request(
        ASR_SUBMIT, data=body,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "X-DashScope-Async": "enable"})
    tid = json.load(urllib.request.urlopen(req, timeout=120))["output"]["task_id"]
    s = None
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
    if s["output"]["task_status"] != "SUCCEEDED":
        print("  ASR task:", s["output"]["task_status"])
        return []
    tr = json.load(urllib.request.urlopen(
        s["output"]["results"][0]["transcription_url"], timeout=60))["transcripts"][0]
    return [(x["begin_time"] / 1000.0, x["end_time"] / 1000.0, x["text"])
            for x in tr.get("sentences", [])]


def map_lines(lines, sents, part_dur, char_offset=0):
    """Distribute ASR sentence timings over our script lines by char ratio."""
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


# ------------------------------------------------------------------ BGM / mix
def build_bed(target_sec):
    """Loop the instrumental fragments (or the old bgm.wav) into a bed."""
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
                 os.path.join(BACKUP, "bgm.wav")]
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
    """Per-block gain: duck_gain under speech, 1.0 in the gaps.

    Returns (gain, speech_mask). The mask is the gate the gain was built from,
    so the level report can measure the duck over the *silent* stretches of the
    audio instead of the caption timings, which are offset from the waveform.
    """
    mono = narr.mean(1)
    n = len(mono) // WIN
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    # Gate the duck on the actual speech envelope. -25 dB below the speech level
    # sits inside the breath/room-tone range, so the gate stayed open and the
    # BGM was ducked 97% of the runtime -- it never swelled in the gaps, which
    # is the opposite of what was asked for. -15 dB tracks the speech itself.
    active = db > (speech_db - 15.0)
    tgt_gain = 10 ** (-DUCK_DB / 20.0)
    att = 1 - math.exp(-0.01 / 0.02)   # ~20 ms attack
    rel = 1 - math.exp(-0.01 / 0.40)   # ~400 ms release, so gaps actually swell
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


def analyze(narr, ducked, scenes, g_full=None, active=None):
    sp = []
    for s in scenes:
        base = s["startFrames"] / FPS
        for ln in s["lines"]:
            a = base + ln["start"]
            sp.append((a, a + ln["dur"]))
    sp.sort()
    gaps = []
    prev = 0.0
    for a, b in sp:
        if a - prev > 0.35:
            gaps.append((prev + 0.15, a - 0.15))
        prev = max(prev, b)
    total = len(narr) / SR
    if total - prev > 0.35:
        gaps.append((prev + 0.15, total))

    def avg_db(arr, wins):
        """Mean level of the windows, averaged in the power domain.

        Averaging dB values directly lets a couple of digital-silence windows
        (-180 dB) drag the whole figure down by tens of dB, which is what made
        the first v5 level report unreadable.
        """
        vals = []
        for a, b in wins:
            i, j = int(a * SR), int(b * SR)
            if j - i > SR // 10:
                vals.append(rms_db(arr[i:j]))
        if not vals:
            return float("nan")
        p = sum(10 ** (v / 10.0) for v in vals) / len(vals)
        return 10 * math.log10(max(p, 1e-18))

    ns = avg_db(narr, sp)
    dg = avg_db(ducked, gaps)
    ds = avg_db(ducked, sp)
    print(f"[levels] narration speech {ns:6.1f} dB | "
          f"BGM in gaps {dg:6.1f} dB | BGM under speech {ds:6.1f} dB | "
          f"duck {dg - ds:4.1f} dB")

    if g_full is not None:
        gb = g_full[::WIN]

        def gmean(wins):
            idx = []
            for a, b in wins:
                i, j = int(a * SR) // WIN, int(b * SR) // WIN
                idx.extend(range(max(0, i), min(len(gb), j)))
            return float(np.mean(gb[idx])) if idx else float("nan")

        gs, gg = gmean(sp), gmean(gaps)
        print(f"  duck gain applied: under speech {gs:.3f} | in gaps {gg:.3f} "
              f"({20 * math.log10(max(gg, 1e-9) / max(gs, 1e-9)):+.1f} dB swell)")

    if active is not None:
        # The caption timings drift from the waveform (the ASR's sentence ends
        # overshoot the part), so the windows above can sit inside speech and
        # understate the duck. The gate is the truth: measure over the blocks
        # it closed, past the release ramp.
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
                m_s[i0 + 20: j0 - 20] = True          # skip the 200 ms ramps
        silent = [(i0, j0) for i0, j0 in runs(False)
                  if (j0 - i0) * WIN / SR > 0.5]
        m_g = np.zeros(nb, bool)
        for i0, j0 in silent:
            m_g[i0 + 45: j0] = True                   # past the 400 ms release
        ns2, bs, bg = blk_db(narr, m_s), blk_db(ducked, m_s), blk_db(ducked, m_g)
        print(f"  duck (gate): narration {ns2:6.1f} dB | BGM under speech "
              f"{bs:6.1f} dB | BGM in silent gaps {bg:6.1f} dB | duck "
              f"{bs - bg:4.1f} dB over "
              f"{sum(j0 - i0 for i0, j0 in silent) * WIN / SR:.1f}s of silence")


# ----------------------------------------------------------------------- main
def main():
    os.makedirs(V5, exist_ok=True)
    os.makedirs(PUBLIC_AUDIO, exist_ok=True)
    force = "--force" in sys.argv
    key = load_key()
    script = json.load(open(SCRIPT, encoding="utf-8"))
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}

    # keep the previous timings for reference
    if os.path.exists(TIMINGS) and not os.path.exists(os.path.join(V5, "timings_v4_backup.json")):
        shutil.copy(TIMINGS, os.path.join(V5, "timings_v4_backup.json"))

    groups = [script[:5], script[5:]]
    scenes = []
    part_start, part_span_end, part_dur, part_wav = {}, {}, {}, {}
    cursor_frames = 0

    for gi, g in enumerate(groups):
        name = f"part{gi + 1}"
        text = "".join("".join(s["lines"]) for s in g)
        wav = os.path.join(V5, name + ".wav")
        if not force and name in cache and os.path.exists(wav):
            dur, sents = cache[name]["dur"], cache[name]["sents"]
            print(f"=== {name} cached: {dur:.2f}s, {len(sents)} ASR sentences ===")
        else:
            print(f"=== {name}: {g[0]['id']}..{g[-1]['id']}  {len(text)} chars -> synth "
                  f"(voice {VOICE}) ===")
            wav, url, api_dur = synth_part(text, name, key)
            dur = float(api_dur) if api_dur else dur_of(wav)
            rate = len(text) / dur
            print(f"    audio {dur:.2f}s (api {api_dur}s, file {dur_of(wav):.2f}s)"
                  f"  {rate:.2f} chars/s = {rate * 60:.0f} 字/分")
            if rate < 2.6 or rate > 6.5:
                print("    !! WARNING: unusual speech rate, check for truncation")
            sents = asr_sentences(url, key)
            heard = "".join(x[2] for x in sents)
            print(f"    ASR: {len(sents)} sentences, {len(heard)} chars "
                  f"(script {len(text)})")
            if len(heard) < 0.75 * len(text):
                print("    !! WARNING: ASR heard much less than the script")
            print(f"    heard: {heard[:140]}")
            cache[name] = {"dur": dur, "sents": sents, "url": url, "voice": VOICE}
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        part_dur[name] = dur
        part_wav[name] = wav          # pitch-aligned below, once both parts are in

        start_sec = 0.0 if gi == 0 else cursor_frames / FPS + PART_GAP
        part_start[name] = start_sec
        cursor_frames = int(round(start_sec * FPS))

        char_off, first = 0, True
        for sc in g:
            lines, sc_dur_end = map_lines(sc["lines"], sents, dur, char_off)
            sc_start = 0.0 if first else max(0.0, lines[0]["start"] - LEAD)
            sc_dur = max(1.0, sc_dur_end - sc_start)
            frames = int(round(sc_dur * FPS))
            rel = [{"text": x["text"],
                    "start": round(x["start"] - sc_start, 3),
                    "dur": x["dur"]} for x in lines]
            scenes.append({
                "id": sc["id"], "name": sc["name"],
                "audio": "audio/final_mix.wav",
                "audioOffset": round(cursor_frames / FPS, 3),
                "title": sc["title"], "sub": sc["sub"],
                "kind": sc["kind"], "visual": sc["visual"],
                "dur": round(sc_dur, 3), "durFrames": frames,
                "startFrames": cursor_frames, "lines": rel,
            })
            print(f"    {sc['id']} start={cursor_frames / FPS:7.2f}s "
                  f"dur={sc_dur:6.2f}s ({frames}f)")
            cursor_frames += frames
            char_off += sum(len(x) for x in sc["lines"])
            first = False
        part_span_end[name] = cursor_frames / FPS

    total_frames = cursor_frames
    total_sec = total_frames / FPS

    # ---- one pitch for the whole narration (everything below reads the result)
    part_wav = align_pitch(part_wav)

    # ---- loudness-match parts, build the narration timeline
    levels = {n: speech_rms_db(read_wav_f(part_wav[n]))
              for n in ("part1", "part2")}
    target = sum(levels.values()) / 2
    for n in ("part1", "part2"):
        a = read_wav_f(part_wav[n])
        gain = 10 ** ((target - levels[n]) / 20.0)
        print(f"  {n}: speech {levels[n]:.1f} dB -> {target:.1f} dB "
              f"({20 * math.log10(gain):+.1f} dB)")
        write_wav_f(os.path.join(V5, n + "_norm.wav"), a * gain)

    buf_len = int((total_sec + 0.25) * SR)
    narr = np.zeros((buf_len, 2), np.float32)
    for gi, g in enumerate(groups):
        name = f"part{gi + 1}"
        a = read_wav_f(os.path.join(V5, name + "_norm.wav"))  # loudness-matched
        off = int(round(part_start[name] * SR))
        keep = int(round((part_span_end[name] - part_start[name]) * SR))
        seg = a[:keep] if len(a) > keep else a
        f = int(0.03 * SR)
        if len(seg) > 2 * f:
            seg = seg.copy()
            seg[:f] *= np.linspace(0, 1, f)[:, None]
            seg[-f:] *= np.linspace(1, 0, f)[:, None]
        end = min(buf_len, off + len(seg))
        narr[off:end] = seg[: end - off]
    write_wav_f(os.path.join(V5, "narration_full.wav"), narr)

    # ---- BGM bed, scaled 8 dB below the speech level
    bed = build_bed(total_sec + 0.5)
    bed = bed[:buf_len] if len(bed) >= buf_len else np.concatenate(
        [bed, np.zeros((buf_len - len(bed), 2), np.float32)])
    bed_db = rms_db(bed)
    bed *= 10 ** ((target + BGM_GAP_DB - bed_db) / 20.0)
    print(f"  bed level: {bed_db:.1f} dB -> {target + BGM_GAP_DB:.1f} dB "
          f"(gaps, speech {target:.1f} dB)")

    # ---- duck under speech and mix
    g_full, active = duck_curve(narr, target)
    print(f"  duck curve: gain {g_full.min():.3f}..{g_full.max():.3f}, "
          f"ducked {float((g_full < 0.9).mean()) * 100:.0f}% of the time")
    ducked = bed * g_full[:, None]
    mix = narr + ducked
    peak = float(np.abs(mix).max())
    if peak > 0.97:
        print(f"  limiting mix: peak {peak:.3f} -> 0.97")
        mix *= 0.97 / peak
    fo = int(1.2 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]

    write_wav_f(os.path.join(V5, "ducked_bgm.wav"), ducked)
    final = os.path.join(PUBLIC_AUDIO, "final_mix.wav")
    write_wav_f(final, mix)
    print(f"\n  final_mix.wav: {len(mix) / SR:.2f}s, "
          f"{os.path.getsize(final) // (1024 * 1024)} MB")
    analyze(narr, ducked, scenes, g_full, active)

    # ---- mp3 preview for quick listening
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    mp3 = os.path.join(V5, "final_mix.mp3")
    subprocess.run([exe, "-y", "-i", final, "-c:a", "libmp3lame",
                    "-b:a", "192k", mp3], capture_output=True)
    print("  mp3 preview:", mp3)

    # ---- timings for Remotion
    json.dump({"fps": FPS, "durationInFrames": total_frames, "scenes": scenes},
              open(TIMINGS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  timings.json: {total_frames} frames = {total_sec:.2f}s")

    # ---- remove the old sounds from the project (kept in a backup dir)
    os.makedirs(BACKUP, exist_ok=True)
    for fn in ("part1.wav", "part1_std.wav", "part2.wav", "part2_std.wav", "bgm.wav"):
        src = os.path.join(PUBLIC_AUDIO, fn)
        if os.path.exists(src):
            dst = os.path.join(BACKUP, fn)
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            print("  moved out of public/:", fn)
    print("  public/audio now:", sorted(os.listdir(PUBLIC_AUDIO)))
    print("\ndone. Main.tsx already plays audio/final_mix.wav; re-render next.")


if __name__ == "__main__":
    main()
