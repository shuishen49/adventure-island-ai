"""Build a low-volume background-music bed for the video.

The TTS model cannot emit a music-only track (a "no speech" request hangs),
but every music-flavoured render starts with ~0.4-1.7 s of pure instrumental
intro and ends with a tail. Those clean fragments are spliced together,
cross-faded, and looped to cover the whole narration.
"""
import glob
import os
import sys

import numpy as np
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000


def read_wav(path):
    w = wave.open(path)
    n = w.getnframes()
    d = np.frombuffer(w.readframes(n), dtype=np.int16)
    return d.reshape(-1, w.getnchannels()).astype(np.float32) / 32768.0


def write_wav(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())


def envelope(mono, seg=960):
    n = len(mono) // seg
    return np.array([
        20 * np.log10(max(np.sqrt((mono[i * seg:(i + 1) * seg] ** 2).mean()), 1e-9))
        for i in range(n)
    ])


def music_parts(path):
    """Return the intro + tail segments that contain no speech."""
    a = read_wav(path)
    mono = a.mean(1)
    db = envelope(mono)
    loud = np.where(db > -30)[0]
    if len(loud) == 0:
        return []
    s = loud[0] * 960
    e = (loud[-1] + 1) * 960
    out = []
    if s > SR // 4:                      # intro long enough to be usable
        out.append(a[:s])
    if len(a) - e > SR // 4:             # tail
        out.append(a[e:])
    return out


def crossfade(x, y, n=SR // 2):
    n = min(n, len(x), len(y))
    if n <= 0:
        return np.concatenate([x, y])
    f = np.linspace(0, 1, n)[:, None]
    head = x[:-n] if len(x) > n else np.zeros((0, 2), np.float32)
    tail_join = x[-n:] * (1 - f) + y[:n] * f
    rest = y[n:]
    return np.concatenate([head, tail_join, rest])


def main():
    srcs = sorted(glob.glob(os.path.join(ROOT, "outputs", "tts", "*.wav")))
    parts = []
    for p in srcs:
        name = os.path.basename(p)
        if not (name.startswith("bg_") or name.startswith("rate_")):
            continue
        got = music_parts(p)
        for g in got:
            if len(g) > SR // 4:
                parts.append(g)
        print(f"{name}: {len(got)} fragment(s)")

    if not parts:
        print("no music fragments found")
        return

    bed = parts[0]
    for p in parts[1:]:
        bed = crossfade(bed, p)
    print(f"raw bed: {len(bed) / SR:.2f}s from {len(parts)} fragments")

    # make it seamless by fading the seam, then loop to the needed length
    target = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    loop = bed
    fade = min(SR, len(bed) // 4)
    head = bed[:fade]
    tail = bed[-fade:]
    seam = head * np.linspace(1, 0, fade)[:, None] + tail * np.linspace(0, 1, fade)[:, None]
    loop = np.concatenate([bed[fade:-fade], seam])

    reps = int(np.ceil(target / (len(loop) / SR))) + 1
    full = np.concatenate([loop] * reps)[: int(target * SR)]

    # gentle overall fade in/out
    fi = int(2.0 * SR)
    fo = int(3.0 * SR)
    full[:fi] *= np.linspace(0, 1, fi)[:, None]
    full[-fo:] *= np.linspace(1, 0, fo)[:, None]

    full *= 0.075          # ~ -22 dB, stays well under the narration
    out = os.path.join(ROOT, "video", "public", "audio", "bgm.wav")
    write_wav(out, full)
    print(f"wrote {out}: {len(full) / SR:.2f}s")


if __name__ == "__main__":
    main()
