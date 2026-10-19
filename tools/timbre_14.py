"""Measure and match the timbre of the 8 narration parts.

The delivered narration is 8 independent synthesis requests, so every part is
its own take: the level and F0 are already matched (see verify_14_speech.py),
but the spectral envelope still moves from part to part, which is what makes
the film sound like several narrators.

This tool:

  probe  - report the bark-band long-term average spectrum (LTAS) of each part
           and how far each one sits from a reference
  match  - write a timbre-matched copy of every part, all nudged onto one
           common LTAS target, then re-normalise the level

The correction is a smooth, slowly varying magnitude response (24 bark bands,
clipped and smoothed), so it fixes the broad spectral balance without
touching pitch, timing, or the individual words.
"""

import os
import sys
import json
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_14 as b  # noqa: E402  (reuses SR, loud_db, the wav reader, ffmpeg)

V = b.V14
OUT = os.path.join(V, "timbre")
N_FFT = 2048
HOP = 512
N_BAND = 24
F_LO = 70.0
F_HI = 11000.0
MAX_CUT_DB = 14.0
MAX_BOOST_DB = 4.5
# Above 3 kHz a boost would mostly amplify the synthesiser's own noise, so the
# high bands are allowed to be cut harder than they can be lifted.
HF_HZ = 3000.0
HF_MAX_BOOST_DB = 3.0
SMOOTH = 1  # neighbouring-band averaging passes

TMP = os.path.join(V, "_timbre_tmp.wav")


def mono(path):
    a = b.read_wav_f(path)
    if a.ndim > 1:
        a = a.mean(1)
    return a.astype(np.float64)


def bark(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def band_edges():
    e = np.geomspace(F_LO, F_HI, N_BAND + 1)
    return e[:-1], e[1:]


def ltas(a, sr=b.SR):
    """Bark-band long-term average spectrum over every speech frame.

    The early version averaged only the loudest 45% of frames. Those are the
    vowels: they carry almost no energy above 4 kHz, so the sibilants - which
    are quiet in absolute terms but dominate 4-10 kHz - were thrown away, and
    the 6.5 kHz reading swung by 10 dB depending on how loud the take happened
    to be. Averaging all frames (minus the near-silent 5%) is what tracks the
    timbre you actually hear.
    """
    w = np.hanning(N_FFT)
    n = (len(a) - N_FFT) // HOP
    if n < 1:
        return np.zeros(N_BAND)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = a[idx] * w
    sp = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freq = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    en = (frames ** 2).sum(1)
    sp = sp[en >= np.percentile(en, 5)]
    lo, hi = band_edges()
    out = np.empty(N_BAND)
    for i in range(N_BAND):
        m = (freq >= lo[i]) & (freq < hi[i])
        out[i] = sp[:, m].mean()
    return 10 * np.log10(np.maximum(out, 1e-12))


def curve_from(gain_db):
    """Interpolate per-band gains into a smooth magnitude response."""
    lo, hi = band_edges()
    ctr = np.sqrt(lo * hi)
    freq = np.fft.rfftfreq(N_FFT, 1.0 / b.SR)
    x = np.concatenate(([0.0], bark(ctr), [bark(b.SR / 2)]))
    y = np.concatenate(([gain_db[0]], gain_db, [gain_db[-1]]))
    xf = np.clip(bark(np.maximum(freq, 1.0)), x[0], x[-1])
    g = np.interp(xf, x, y)
    return np.interp(freq, freq, g)  # identity; keeps shape explicit


def apply_curve(a, gain_db):
    """Overlap-add filtering with a zero-phase smooth magnitude response."""
    lin = 10 ** (curve_from(gain_db) / 20.0)
    w = np.hanning(N_FFT)
    n = (len(a) - N_FFT) // HOP + 1
    pad = N_FFT + HOP * (n - 1)
    x = np.zeros(pad + N_FFT)
    x[: len(a)] = a
    y = np.zeros_like(x)
    for i in range(n):
        s = i * HOP
        blk = x[s : s + N_FFT]
        sp = np.fft.rfft(blk * w)
        y[s : s + N_FFT] += np.fft.irfft(sp * lin, N_FFT) * w
    return y[: len(a)]


def smooth_clip(g):
    for _ in range(SMOOTH):
        k = np.array([0.25, 0.5, 0.25])
        g = np.convolve(np.concatenate(([g[0]], g, [g[-1]])), k, mode="same")[1:-1]
    g = np.clip(g, -MAX_CUT_DB, MAX_BOOST_DB)
    ctr = np.sqrt(band_edges()[0] * band_edges()[1])
    hi = ctr > HF_HZ
    if hi.any():
        g[hi] = np.minimum(g[hi], HF_MAX_BOOST_DB)
    return g


def part_names():
    return ["part%d" % i for i in range(1, 9)]


def load_parts():
    ps, aus = [], []
    for nm in part_names():
        p = os.path.join(V, nm + "_norm.wav")
        if not os.path.exists(p):
            continue
        ps.append(nm)
        aus.append(mono(p))
    return ps, aus


def probe():
    ps, aus = load_parts()
    L = np.array([ltas(a) for a in aus])
    ref = L[0]
    print("band deviation vs %s (dB), rows = parts" % ps[0])
    print("        " + "".join("%6d" % int(f) for f in np.sqrt(band_edges()[0] * band_edges()[1])[::3]))
    for i, nm in enumerate(ps):
        d = L[i] - ref
        print("%-7s " % nm + "".join("%6.1f" % v for v in d[::3])
              + "   rms %.2f  max %.2f" % (np.sqrt((d ** 2).mean()), np.abs(d).max()))
    # how far each part is from the mean of all parts
    mean = L.mean(0)
    print("\ndistance to the ensemble mean (rms dB over 24 bands):")
    for i, nm in enumerate(ps):
        d = L[i] - mean
        print("  %-7s rms %.2f  max %.2f" % (nm, np.sqrt((d ** 2).mean()), np.abs(d).max()))
    print("\nspread across parts: rms %.2f dB  (band max-min %.2f dB)"
          % (np.sqrt(((L - mean) ** 2).mean()), (L.max(0) - L.min(0)).max()))
    return ps, aus, L


def pass_gains(L, target):
    return [smooth_clip(target - L[i]) for i in range(len(L))]


def spread(L):
    d = L - L.mean(0)
    return np.sqrt((d ** 2).mean())


def match():
    os.makedirs(OUT, exist_ok=True)
    ps, aus = load_parts()
    L = np.array([ltas(a) for a in aus])
    # The target sits at the 30th percentile, i.e. towards the DARK end of the
    # eight takes. Matching to the median would need +13 dB of lift on the dark
    # takes, and a lift up there only raises the synthesiser's own low-level
    # hash. Cutting the bright takes down to the dark ones is both safer and
    # cheaper: the residual after clipping is what you would otherwise hear.
    target = np.percentile(L, 30, axis=0)
    print("target = per-band p30 of %d parts (dB):" % len(ps))
    print("  " + "".join("%6.1f" % v for v in target[::3]))

    cur = list(aus)
    # two passes: the smoothing and the clipped extremes always leave a
    # residual, and the second pass cleans most of it up
    for p in range(2):
        gs = pass_gains(L, target)
        if p == 0:
            for i, nm in enumerate(ps):
                print("  %-7s gain %+5.1f..%+5.1f dB  (rms %.2f)"
                      % (nm, gs[i].min(), gs[i].max(), np.sqrt((gs[i] ** 2).mean())))
        nxt = []
        for i, y in enumerate(cur):
            nxt.append(apply_curve(y, gs[i]))
        cur = nxt
        L = np.array([ltas(a) for a in cur])
        print("  pass %d: spread rms %.2f dB" % (p + 1, spread(L)))

    # The shape correction moves the level too, so put each part back on the
    # loudness it was mixed at: one broadband gain per part, closed-loop on the
    # energy-domain loudness (a straight gain lands off by a fraction of a dB
    # because the loudness measure is gated).
    for i, nm in enumerate(ps):
        want = b.loud_db(np.stack([aus[i], aus[i]], 1))
        g = 0.0
        for _ in range(4):
            y = cur[i] * 10 ** (g / 20.0)
            err = want - b.loud_db(np.stack([y, y], 1))
            if abs(err) < 0.03:
                break
            g += err
        cur[i] = y
        print("  %-7s level %+.2f dB -> %.2f dB" % (nm, g, b.loud_db(np.stack([y, y], 1))))
    L = np.array([ltas(a) for a in cur])
    for i, nm in enumerate(ps):
        dst = os.path.join(OUT, nm + "_norm.wav")
        b.write_wav_f(dst, np.stack([cur[i], cur[i]], 1))
        d = L[i] - np.median(L, axis=0)
        print("  %-7s residual rms %.2f  max %.2f" % (nm, np.sqrt((d ** 2).mean()), np.abs(d).max()))
    ctr = np.sqrt(band_edges()[0] * band_edges()[1])
    sel = (ctr >= 120) & (ctr <= 8000)
    L0 = np.array([ltas(a) for a in aus])
    print("\nvoice band 120 Hz-8 kHz spread: rms %.2f -> %.2f dB"
          % (np.sqrt(((L0[:, sel] - L0[:, sel].mean(0)) ** 2).mean()),
             np.sqrt(((L[:, sel] - L[:, sel].mean(0)) ** 2).mean())))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "probe":
        probe()
    elif cmd == "match":
        match()
    else:
        print("usage: timbre_14.py [probe|match]")
