"""Speech-only check of the two narration halves.

verify_v6_mp4.py read the whole timeline, and the BGM that comes up in the pauses
poisoned both numbers: it dragged the block mean down (music-only blocks sit at
-31..-39 dB, above the -45 dB gate) and it fed music into Praat's pitch tracker.
This gates at -25 dB instead -- speech sits near -16 dB -- so level and pitch are
read from voiced speech alone. Same ruler on the source takes, the placed mix and
the delivered AAC track, so a stale or wrong file shows up as a mismatch.

Run:  python tools/verify_v6_speech.py
"""
import math
import os
import wave

import numpy as np

WIN = 480                     # 10 ms blocks at 48 kHz
GATE = -25.0                  # dB; speech ~-16, BGM-only pauses -31..-39
H1 = (0.20, 41.40)            # part1 speech on the timeline
H2 = (42.70, 86.60)           # part2 speech on the timeline
V6 = r"D:\source_code\adventure-island-ai\outputs\v6"


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
    print(f"  {label:<34} speech {lvl:7.2f} dB | F0 {hz:6.1f} Hz | "
          f"speech blocks {share:4.0f}%")
    return lvl, hz


def pair(lv1, hz1, lv2, hz2):
    print(f"  {'-> part1 vs part2':<34} level {lv1 - lv2:+.2f} dB | "
          f"pitch {12 * math.log2(hz2 / hz1):+.2f} st")


def main():
    print("source takes (no BGM):")
    l1, f1 = measure(os.path.join(V6, "part1_norm.wav"), "part1_norm.wav")
    l2, f2 = measure(os.path.join(V6, "part2_norm.wav"), "part2_norm.wav")
    pair(l1, f1, l2, f2)

    print("placed mix, speech windows:")
    l1, f1 = measure(os.path.join(V6, "final_mix.wav"), "final_mix.wav H1", H1)
    l2, f2 = measure(os.path.join(V6, "final_mix.wav"), "final_mix.wav H2", H2)
    pair(l1, f1, l2, f2)

    print("delivered mp4 audio, speech windows:")
    p = os.path.join(V6, "mp4_audio_check.wav")
    l1, f1 = measure(p, "mp4 audio H1", H1)
    l2, f2 = measure(p, "mp4 audio H2", H2)
    pair(l1, f1, l2, f2)


if __name__ == "__main__":
    main()
