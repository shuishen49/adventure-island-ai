"""Independent check of the delivered mp4: decode its audio, then re-measure the
two things the user asked for -- one consistent voice across the whole film, and
BGM that sits under the speech. Reads the AAC track, not the source wav, so a
stale render or a wrong audio file shows up here.

Run:  python tools/verify_v6_mp4.py outputs/v6/mp4_audio_check.wav
"""
import math
import sys
import wave

import numpy as np

WIN = 480                     # 10 ms blocks at 48 kHz, same ruler as build_v6
H1 = (0.20, 41.40)            # part1 speech: first line start .. last line end
H2 = (42.70, 86.60)           # part2 speech: ditto


def read_wav_f(path):
    w = wave.open(path, "rb")
    ch, sw, sr, n = (w.getnchannels(), w.getsampwidth(),
                     w.getframerate(), w.getnframes())
    a = np.frombuffer(w.readframes(n),
                      dtype="<i2" if sw == 2 else "<i4").astype(np.float32)
    a = a.reshape(-1, ch) / (32768.0 if sw == 2 else 2147483648.0)
    w.close()
    return a, sr


def speech_rms_db(a):
    """Mean 10 ms block level above -45 dB -- build_v6's own ruler."""
    mono = a.mean(1)
    n = len(mono) // WIN
    if n == 0:
        return -99.0
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    loud = db[db > -45.0]
    return float(loud.mean())


def f0_hz(a, sr):
    import parselmouth
    snd = parselmouth.Sound(a.mean(1).astype(np.float64), sr)
    p = snd.to_pitch(time_step=0.01)
    v = p.selected_array["frequency"]
    v = v[v > 0]
    return float(np.median(v)) if len(v) else 0.0


def main(path):
    a, sr = read_wav_f(path)
    print(f"{path}\n  {len(a) / sr:.2f}s @ {sr} Hz, {a.shape[1]}ch, "
          f"peak {float(np.abs(a).max()):.3f}, "
          f"full-scale samples {int((np.abs(a) >= 0.999).sum())}")
    lv, hz = {}, {}
    for name, (t0, t1) in (("part1", H1), ("part2", H2)):
        seg = a[int(t0 * sr):int(t1 * sr)]
        lv[name] = speech_rms_db(seg)
        hz[name] = f0_hz(seg, sr)
        print(f"  {name}: speech {lv[name]:.2f} dB | F0 {hz[name]:.1f} Hz")
    print(f"  -> level difference {lv['part1'] - lv['part2']:+.2f} dB | "
          f"pitch difference {12 * math.log2(hz['part2'] / hz['part1']):+.2f} "
          f"semitones")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/v6/mp4_audio_check.wav")
