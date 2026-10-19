"""Rebuild a part's cache entry, optionally re-timed onto a different take.

Two jobs, one mechanism.

1. part3 and part6 are missing from v14_cache.json. They were deleted to force
   a re-roll, the free tier ran out mid-re-roll, and the entries were never
   written back. Without them build_14 tries to synthesise and dies on a 403.

2. part3's take was picked by pitch alone. It came back at 149 Hz and had to be
   pushed down 3.19 semitones to sit with the rest of the narration. Rubberband
   keeps the formants, but a shift that big still costs harmonicity: the
   segment measures 10.6 dB HNR against 15-18 dB for the parts around it, which
   is what "the voice changes at 1:29" sounds like. The seed sweep left us a
   take of the same text at 120.4 Hz that needs no correction at all.

The replacement is 4.7 s longer, and the model did not simply speak slower:
measured against the old take, the phrasing wanders up to 1 s away from a
uniform stretch, so scaling the timestamps would put every scene in the part
off by up to a second. The timestamps are re-anchored instead:

  1. lines are located in the new recording by correlation - a line is about
     2 s, long enough to lock onto, where a single word (200 ms) is not;
  2. those line boundaries become the anchors of a monotone piecewise-linear
     map from old time to new time;
  3. every stored timestamp is pushed through that map.

Because the stored timestamps are line-sized, the line is the word: the
pipeline's word -> line aggregation then reads them straight back out. The
word-level detail inside a line is not recoverable from the delivered
timings, and nothing downstream uses it.

Usage:
    retime_14.py restore            # rebuild every missing part from timings
    retime_14.py part3 seed_1       # also swap part3 onto a different take
"""

import bisect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_14 as b  # noqa: E402
import remix_14 as r  # noqa: E402

CACHE = b.CACHE
SR = b.SR
WIN = int(0.02 * SR)
TIMINGS = b.TIMINGS


def envelope(x):
    n = len(x) // WIN
    return np.sqrt((x[: n * WIN].reshape(n, WIN) ** 2).mean(1))


def trimmed(path):
    a = b.read_wav_f(path)
    cuts, _ = b.trim_plan(a, b.speech_rms_db(a))
    return b.apply_trim(a, cuts).mean(1).astype(np.float64), cuts


def inverse_cutmap(cuts, raw_sec):
    """trimmed time -> raw time, by binary search on the forward map."""
    cm = b.cut_map(cuts)

    def inv(t):
        lo, hi = 0.0, float(raw_sec)
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if cm(mid) < t:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    return inv


def piecewise(anchors):
    """Monotone piecewise-linear map through (old, new) anchors."""
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    for i in range(1, len(ys)):          # a bad correlation must not run it back
        if ys[i] <= ys[i - 1]:
            ys[i] = ys[i - 1] + 1e-3

    def f(t):
        if t <= xs[0]:
            return ys[0] + (t - xs[0])
        if t >= xs[-1]:
            return ys[-1] + (t - xs[-1])
        j = bisect.bisect_right(xs, t)
        x0, x1 = xs[j - 1], xs[j]
        y0, y1 = ys[j - 1], ys[j]
        return y0 + (y1 - y0) * (t - x0) / max(x1 - x0, 1e-9)

    return f


MFCC_HOP = 0.01
POOL = 4          # 40 ms per alignment frame: plenty for line timing, and it
                  # keeps the DP inside 10^6 cells instead of 10^7


N_FFT = 1024
N_BAND = 26


def _melbank():
    f = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    hz2mel = lambda x: 2595.0 * np.log10(1.0 + x / 700.0)  # noqa: E731
    mel = hz2mel(np.maximum(f, 1.0))
    pts = np.linspace(hz2mel(80.0), hz2mel(7600.0), N_BAND + 2)
    fb = np.zeros((N_BAND, len(f)))
    for i in range(N_BAND):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (mel >= lo) & (mel <= ce)
        dn = (mel >= ce) & (mel <= hi)
        fb[i, up] = (mel[up] - lo) / (ce - lo)
        fb[i, dn] = (hi - mel[dn]) / (hi - ce)
    return fb


_FB = None


def mfcc(path):
    """Log-mel energies, pooled to POOL frames, per-frame mean removed.

    Amplitude envelopes were tried first and are useless here: two different
    phrases have envelopes that look alike, so a 2 s window locks onto the
    wrong place and consecutive lines came back 4-6 s apart. A mel spectrum
    separates the phonemes, which is what makes the alignment hold. The per
    frame mean is subtracted because level is exactly what must not drive the
    match.
    """
    global _FB
    if _FB is None:
        _FB = _melbank()
    a = b.read_wav_f(path).mean(1).astype(np.float64)
    hop = int(round(MFCC_HOP * SR))
    n = (len(a) - N_FFT) // hop
    idx = np.arange(N_FFT)[None, :] + hop * np.arange(n)[:, None]
    sp = np.abs(np.fft.rfft(a[idx] * np.hanning(N_FFT), axis=1)) ** 2
    L = 10.0 * np.log10(np.maximum(sp @ _FB.T, 1e-10))
    L = L - L.mean(1, keepdims=True)
    m = L
    k = len(m) // POOL * POOL
    m = m[:k].reshape(k // POOL, POOL, m.shape[1]).mean(1)
    t = (np.arange(len(m)) + 0.5) * MFCC_HOP * POOL
    return m, t


def dtw_path(X, Y):
    """Banded-free DTW between two MFCC sequences; returns the frame path."""
    n, m = len(X), len(Y)
    inf = 1e18
    D = np.full((n + 1, m + 1), inf, np.float64)
    P = np.zeros((n + 1, m + 1), np.int8)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        diff = Y - X[i - 1]
        dsq = (diff * diff).sum(1)
        row = D[i]
        prow = D[i - 1]
        for j in range(1, m + 1):
            a = prow[j - 1]
            b = prow[j]
            c = row[j - 1]
            if a <= b and a <= c:
                row[j] = a + dsq[j - 1]
                P[i, j] = 0
            elif b <= c:
                row[j] = b + dsq[j - 1]
                P[i, j] = 1
            else:
                row[j] = c + dsq[j - 1]
                P[i, j] = 2
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i <= 0:
            j -= 1
            continue
        if j <= 0:
            i -= 1
            continue
        p = P[i, j]
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path


def mfcc_trimmed(path):
    """MFCC of a take with its pauses cut, so the times line up with the rest."""
    a, _cuts = trimmed(path)
    tmp = os.path.join(b.ROLLS, "_mfcc_tmp.wav")
    b.write_wav_f(tmp, np.stack([a, a], 1).astype(np.float32))
    return mfcc(tmp)


def dtw_warp(path, tx, ty):
    """Monotone old-time -> new-time map from a DTW path.

    Several old frames can map to one new frame and vice versa; taking the
    median new time per old frame keeps the result monotone and smooth.
    """
    pairs = {}
    for i, j in path:
        pairs.setdefault(i, []).append(j)
    keys = sorted(pairs)
    xs = tx[keys]
    ys = np.array([ty[int(np.median(pairs[k]))] for k in keys])
    # make it strictly increasing before interpolating
    for k in range(1, len(ys)):
        if ys[k] <= ys[k - 1]:
            ys[k] = ys[k - 1] + 1e-3
    return lambda t: float(np.interp(t, xs, ys))


def part_scenes(name):
    """The scenes belonging to one part, straight from the packing.

    Do not try to recover this from timings.json: a part's first scene can
    start before the part's own origin (the layout subtracts a lead-in), so
    bucketing scenes by absolute time puts the boundary scenes in the wrong
    part - part3 and part6 were each off by one scene that way.
    """
    script = json.load(open(b.SCRIPT, encoding="utf-8"))
    gs = b.pack_parts(script)
    names = ["part%d" % (i + 1) for i in range(len(gs))]
    return [sc["id"] for sc in gs[names.index(name)]]


def part_lines(timings, origins, name):
    """Line timings for one part, as (text, t0, t1) in part-relative time.

    The text stored here is what the ASR heard (digits, no punctuation, "它"
    for "他" in places) - that is what the word stream has always carried, and
    it is what the content alignment is measured against.
    """
    ids = set(part_scenes(name))
    lo = origins[name]
    out = []
    for sc in timings["scenes"]:
        if sc["id"] not in ids:
            continue
        base = sc["startFrames"] / timings["fps"]
        for ln in sc["lines"]:
            out.append((ln["text"], base + ln["start"] - lo,
                        base + ln["start"] + ln["dur"] - lo))
    return out


def build_entry(name, lines, wav, f0, dur):
    return {
        "sha": None,
        "wav": wav,
        "url": "",
        "f0": f0,
        "dur": dur,
        "words": [[s, e, t] for t, s, e in lines],
        "rate": b.RATE,
        "voice": b.VOICE,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "restore"
    cache = json.load(open(CACHE, encoding="utf-8"))
    timings = json.load(open(TIMINGS, encoding="utf-8"))
    origins = r.origins()
    names = r.NAMES
    # the text the pipeline hashes is the script as written, not the ASR
    # transcript: the two differ by every numeral and every comma
    script_text = {
        nm: "".join("".join(sc["lines"]) for sc in g)
        for nm, g in zip(names, b.pack_parts(json.load(open(b.SCRIPT,
                                                           encoding="utf-8"))))}

    missing = [n for n in names if n not in cache.get("parts", {})]
    print("missing from cache:", missing or "none")

    if mode == "restore":
        targets = missing
        swap = None
    else:
        targets = [mode]
        swap = sys.argv[2] if len(sys.argv) > 2 else None

    for name in targets:
        lines = part_lines(timings, origins, name)
        if not lines:
            print(f"  {name}: no scenes found, skipped")
            continue

        if swap:
            old = cache["parts"].get(name, {}).get("wav")
            if not old:
                # the entry was deleted; fall back to the take it shipped with
                old = os.path.join(b.ROLLS, "part3_r8.wav") if name == "part3" \
                    else None
            if not old or not os.path.exists(old):
                print(f"  {name}: no old take to anchor against, skipped")
                continue
            new = os.path.join(b.ROLLS, swap + ".wav")
            # the old side has to be the file the delivered timings were
            # derived from, i.e. the take as it was pitch-corrected
            old_ref = old.replace(".wav", "_aligned.wav")
            if not os.path.exists(old_ref):
                old_ref = old
            ot, oc = trimmed(old_ref)
            nt, nc = trimmed(new)
            X, tx = mfcc_trimmed(old_ref)
            Y, ty = mfcc_trimmed(new)
            path = dtw_path(X, Y)
            warp = dtw_warp(path, tx, ty)
            scale = len(nt) / len(ot)
            # how far the real alignment sits from a straight stretch: this is
            # exactly the error a proportional rescale would have baked in
            grid = np.linspace(0, tx[-1], 60)
            d = np.array([warp(t) - t * scale for t in grid])
            print(f"  {name}: {os.path.basename(old)} -> {swap}  "
                  f"{len(ot) / SR:.2f}s -> {len(nt) / SR:.2f}s")
            print(f"    drift from a uniform stretch: mean {d.mean():+.2f}s "
                  f"max |d| {np.abs(d).max():.2f}s "
                  f"(scaling would have left {np.abs(d).max():.2f}s of error)")

            inv = inverse_cutmap(nc, len(nt) / SR
                                 + sum(e - s for s, e in nc) / SR)
            lines = [(t, warp(t0), warp(t1)) for t, t0, t1 in lines]
            lines = [(t, inv(t0), inv(t1)) for t, t0, t1 in lines]
            wav = new
        else:
            # No swap: the take the film shipped with. The cuts must be read
            # off the PITCH-ALIGNED file, not the raw take: build_14 trims
            # after the shift, and a 3-semitone shift moves the loudness
            # envelope enough to land the cuts elsewhere - which threw the
            # rebuilt scenes 1.2 s off when the raw take was used instead.
            wav = os.path.join(b.ROLLS, {
                "part3": "part3_r8", "part6": "part6_r0"}.get(name, name) + ".wav")
            ref_take = wav.replace(".wav", "_aligned.wav")
            if not os.path.exists(ref_take):
                ref_take = wav
            tr, cuts = trimmed(ref_take)
            raw_sec = len(tr) / SR + sum(e - s for s, e in cuts) / SR
            inv = inverse_cutmap(cuts, raw_sec)
            lines = [(t, inv(t0), inv(t1)) for t, t0, t1 in lines]

        # keep the chain strictly monotone so the line layout stays ordered
        for i in range(1, len(lines)):
            if lines[i][1] < lines[i - 1][1]:
                pad = lines[i - 1][1] + 1e-3 - lines[i][1]
                lines[i] = (lines[i][0], lines[i][1] + pad, lines[i][2] + pad)
            if lines[i][2] <= lines[i][1]:
                lines[i] = (lines[i][0], lines[i][1], lines[i][1] + 0.05)

        f0 = b.measure_f0(wav)
        dur = b.dur_of(wav)
        cache.setdefault("parts", {})[name] = build_entry(name, lines, wav, f0, dur)
        print(f"  {name}: {os.path.basename(wav)} F0 {f0:.1f} Hz {dur:.2f}s "
              f"{len(lines)} lines -> "
              f"{12 * np.log2(f0 / cache['parts']['part1']['f0']):+.2f} st vs part1")

    # pick_roll keys the cache on sha(text, target). Leave it None and it
    # decides the entry is stale and re-synthesises - which with no free tier
    # left turns into six 403s and a dead build.
    ref = cache["parts"]["part1"]["f0"]
    for name in targets:
        cache["parts"][name]["sha"] = b.sha_of(script_text[name], ref)

    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print("cache written (sha keyed to part1 at %.1f Hz)" % ref)


if __name__ == "__main__":
    main()
