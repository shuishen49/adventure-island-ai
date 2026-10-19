"""Rebuild the 1-2..1-4 mix from timbre-matched parts.

The scene layout in timings.json was derived from the audio, so it stays valid
as long as the parts keep their length - which the timbre correction does. This
rebuilds narration_full.wav on the original offsets, keeps the BGM bed and its
ducking curve exactly as they were mixed, and writes a new final_mix.wav.

No re-render needed: feed the new mix to the existing mp4 with -c:v copy.

    python tools/remix_14.py [check]
    check   rebuild from the UNMODIFIED parts and compare against the current
            narration_full.wav - proves the offsets are right before trusting
            the timbre-matched version
"""

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_14 as b  # noqa: E402
import timbre_14 as t  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = b.V14
PUBLIC_AUDIO = os.path.join(ROOT, "video14", "public", "audio")
NAMES = ["part%d" % i for i in range(1, 9)]


def groups_of():
    script = json.load(open(b.SCRIPT, encoding="utf-8"))
    return b.pack_parts(script)


def origins():
    """Absolute start time of every part, recovered from timings.json.

    Scene k of group g starts at origin_g + base - LEAD, and line i of that
    scene sits at origin_g + _s. So the absolute end of a group's last line is
    origin_g + prev_end, and the next part starts PART_GAP after that.
    """
    tim = json.load(open(b.TIMINGS, encoding="utf-8"))
    byid = {s["id"]: s for s in tim["scenes"]}
    grps = groups_of()
    out, origin = {}, 0.0
    for g in grps:
        out[g[0]["id"]] = origin
        last = byid[g[-1]["id"]]
        ln = last["lines"][-1]
        origin = (origin * 0 + (last["startFrames"] / b.FPS + ln["start"] + ln["dur"])
                  + b.PART_GAP)
    # map group -> part name (groups are in the same order as the part names)
    return {NAMES[i]: v for i, v in enumerate(out.values())}


def refine(prot):
    """Snap the offsets to the sample.

    The recovered offsets carry the 1/30 s rounding of the scene starts, up to
    +-16 ms - enough to slide a part against the scene layout it was measured
    from. Fit each part against the master by least squares (a plain dot product
    peaks wherever the loudest window happens to land, which on speech can be
    80+ samples off), coarse then fine, and confirm on the whole part.
    """
    ref = b.read_wav_f(os.path.join(V, "narration_full.wav")).mean(1)
    AT, N = int(3 * b.SR), int(4 * b.SR)
    out = {}
    for nm in NAMES:
        a = b.read_wav_f(os.path.join(V, nm + "_norm.wav")).mean(1)
        base = int(round(prot[nm] * b.SR))
        snip = a[AT: AT + N]

        def fit(lags):
            bl, bv = 0, -1e30
            for lag in lags:
                w = ref[base + AT + lag: base + AT + lag + N]
                v = float(np.dot(snip, w)) - 0.5 * float(np.dot(w, w))
                if v > bv:
                    bv, bl = v, lag
            return bl

        lag = fit(range(-900, 901, 4))
        lag = fit(range(lag - 6, lag + 7))
        out[nm] = (base + lag) / b.SR
        print("  %-7s %.5f -> %.5f (%+d samples)" % (nm, prot[nm], out[nm], lag))
    return out


def build(src_dir):
    narr0 = b.read_wav_f(os.path.join(V, "narration_full.wav"))
    buf = np.zeros_like(narr0)
    prot = refine(origins())
    for nm in NAMES:
        a = b.read_wav_f(os.path.join(src_dir, nm + "_norm.wav"))
        off = int(round(prot[nm] * b.SR))
        f = int(0.03 * b.SR)
        seg = a.copy()
        if len(seg) > 2 * f:
            seg[:f] *= np.linspace(0, 1, f)[:, None]
            seg[-f:] *= np.linspace(1, 0, f)[:, None]
        end = min(len(buf), off + len(seg))
        buf[off:end] = seg[: end - off]
    return buf, prot


def check():
    buf, prot = build(V)
    ref = b.read_wav_f(os.path.join(V, "narration_full.wav"))
    d = buf - ref
    print("offsets (snapped to the sample):")
    for nm in NAMES:
        print("  %-7s %.3f s" % (nm, prot[nm]))
    print("rebuild vs current narration_full: max |diff| %.5f, rms diff %.6f"
          % (float(np.abs(d).max()), float(np.sqrt((d ** 2).mean()))))
    return buf


def remix():
    buf, prot = build(t.OUT)
    ref = b.read_wav_f(os.path.join(V, "narration_full.wav"))[: len(buf)]
    print("offsets: " + " ".join("%s=%.2f" % (nm[4:], prot[nm]) for nm in NAMES))
    print("narration level: old %.2f dB -> new %.2f dB"
          % (b.speech_rms_db(ref), b.speech_rms_db(buf)))

    ducked = b.read_wav_f(os.path.join(V, "ducked_bgm.wav"))
    n = min(len(buf), len(ducked))
    mix = buf[:n] + ducked[:n]
    peak = float(np.abs(mix).max())
    if peak > 0.97:
        print("  limiting mix: peak %.3f -> 0.97 (%.2f dB)"
              % (peak, 20 * np.log10(0.97 / peak)))
        mix *= 0.97 / peak
    fo = int(1.2 * b.SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]
    clipped = int((np.abs(mix) >= 0.999).sum())

    b.write_wav_f(os.path.join(V, "final_mix.wav"), mix)
    final = os.path.join(PUBLIC_AUDIO, "final_mix.wav")
    b.write_wav_f(final, mix)
    print("  final_mix.wav %.2fs, clipped samples %d, peak %.3f, %d MB"
          % (len(mix) / b.SR, clipped, float(np.abs(mix).max()),
             os.path.getsize(final) // (1024 * 1024)))
    return mix


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
    else:
        remix()
