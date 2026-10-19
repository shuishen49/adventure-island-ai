"""Speech-only verification of the 1-2 → 1-4 narration.

Same ruler as verify_v6_speech.py, generalised to N parts:
  - gate at -25 dB, not -45. The BGM (here the game's own music) comes up in
    the pauses at -31..-39 dB; the -45 gate counted those blocks as speech and
    dragged both the level and Praat's F0 estimate.
  - F0 only from voiced frames that are also above the gate.
  - three passes: source takes (no BGM), the placed mix, and the audio
    re-extracted from the delivered mp4. A stale or wrong file shows up as a
    mismatch between the three.

Run:  python tools/verify_14_speech.py [path/to/out.mp4]
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np
import imageio_ffmpeg

WIN = 480                     # 10 ms blocks at 48 kHz
GATE = -25.0                  # dB; speech ~-16, BGM-only pauses -31..-39
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V14 = os.path.join(ROOT, "outputs", "v14")
# the timbre-matched parts are what actually ships, so they are what gets
# measured; falls back to the raw takes if timbre_14.py has not been run
TAKES = os.path.join(V14, "timbre")
if not os.path.isdir(TAKES):
    TAKES = V14
MP4 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "video14", "out", "adventure_island_14.mp4")


def read_wav_f(path):
    w = wave.open(path, "rb")
    ch, sw, sr, n = (w.getnchannels(), w.getsampwidth(),
                     w.getframerate(), w.getnframes())
    a = np.frombuffer(w.readframes(n),
                      dtype="<i2" if sw == 2 else "<i4").astype(np.float32)
    a = a.reshape(-1, ch) / (32768.0 if sw == 2 else 2147483648.0)
    w.close()
    return a, sr


def block_db(a):
    mono = a.mean(1)
    n = len(mono) // WIN
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(e, 1e-9))


def measure(path, label, region=None):
    a, sr = read_wav_f(path)
    if region is not None:
        a = a[int(region[0] * sr):int(region[1] * sr)]
    db = block_db(a)
    loud = db[db > GATE]
    lvl = float(loud.mean()) if len(loud) else -99.0

    import parselmouth
    snd = parselmouth.Sound(a.mean(1).astype(np.float64), sr)
    p = snd.to_pitch(time_step=0.01)
    f = p.selected_array["frequency"]
    try:
        t = np.asarray(p.xs())
    except Exception:
        t = (np.arange(len(f)) + 0.5) * 0.01
    idx = np.clip((t / 0.01).astype(int), 0, len(db) - 1)
    voiced = (f > 0) & (db[idx] > GATE)
    hz = float(np.median(f[voiced])) if voiced.any() else 0.0
    share = 100.0 * len(loud) / max(len(db), 1)
    print(f"  {label:<28} speech {lvl:7.2f} dB | F0 {hz:6.1f} Hz | "
          f"speech blocks {share:4.0f}%")
    return lvl, hz


def parts(d=None):
    out = []
    i = 1
    while os.path.exists(os.path.join(d or TAKES, f"part{i}_norm.wav")):
        out.append(f"part{i}")
        i += 1
    return out


def timbre(d, ps):
    """Bark-band spread of the takes: the part-to-part timbre difference, which
    is what 'several narrators' sounds like. Reported for 120 Hz-8 kHz."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import timbre_14 as t
    ctr = np.sqrt(t.band_edges()[0] * t.band_edges()[1])
    sel = (ctr >= 120) & (ctr <= 8000)
    L = np.array([t.ltas(t.mono(os.path.join(d, n + "_norm.wav"))) for n in ps])
    d0 = L - L.mean(0)
    return np.sqrt((d0[:, sel] ** 2).mean()), np.abs(d0[:, sel]).max()


def spread(vals, unit="dB"):
    if unit == "st":
        vals = [12 * math.log2(v / vals[0]) for v in vals]
        return max(vals) - min(vals)
    return max(vals) - min(vals)


def main():
    ps = parts()
    print(f"source takes (no BGM), {len(ps)} parts, "
          f"from {os.path.relpath(TAKES, ROOT)}:")
    lv, hz = [], []
    for n in ps:
        l, h = measure(os.path.join(TAKES, n + "_norm.wav"), n + "_norm.wav")
        lv.append(l)
        hz.append(h)
    print(f"  -> spread: level {spread(lv):+.2f} dB | "
          f"pitch {spread(hz, 'st'):+.2f} st")
    if TAKES != V14:
        a0, m0 = timbre(V14, ps)
        a1, m1 = timbre(TAKES, ps)
        print(f"  -> timbre 120 Hz-8 kHz: rms {a0:.2f} dB (max {m0:.1f}) "
              f"-> rms {a1:.2f} dB (max {m1:.1f})")

    print("placed mix (narration + ducked game music):")
    l, h = measure(os.path.join(V14, "final_mix.wav"), "final_mix.wav")
    print(f"  -> placed mix speech {l:.2f} dB | F0 {h:.1f} Hz")

    if not os.path.exists(MP4):
        print(f"\n(no delivered mp4 at {MP4})")
        return
    tmp = os.path.join(V14, "mp4_audio_check.wav")
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", MP4, "-vn", "-ac", "2", "-ar", "48000",
                    "-c:a", "pcm_s16le", tmp], check=True)
    print("delivered mp4 audio:")
    a, sr = read_wav_f(tmp)
    db = block_db(a)
    loud = db[db > GATE]
    print(f"  {'mp4 (whole timeline)':<28} speech {float(loud.mean()):7.2f} dB | "
          f"speech blocks {100.0 * len(loud) / max(len(db), 1):4.0f}% | "
          f"{len(a) / sr:.2f}s")


if __name__ == "__main__":
    main()
