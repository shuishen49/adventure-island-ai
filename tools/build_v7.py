"""v7 mix: keep the v6 dry narration, swap the synthesized BGM bed for the
real in-game audio of the NEW clear run (outputs/replay_sound/game_audio.wav,
stretched by the same factor as the footage so game events line up with what
is on screen).

No TTS is re-run here: narration_full.wav (dry voice, 48 kHz stereo) is used
as-is. Bed level / duck gate / limiter / end fade are identical to build_v6.
"""
import math
import os
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000
WIN = 480            # 10 ms envelope block, same as v6
BGM_GAP_DB = -8.0    # bed RMS vs narration reference, same as v6
DUCK_DB = 13.0       # duck depth, same as v6
BED_TRIM_DB = -0.92  # user: overall game-audio bed 10% quieter (20*log10(0.9))

NARR = os.path.join(ROOT, "outputs", "v6", "narration_full.wav")
BED = os.path.join(ROOT, "outputs", "v7", "bed_src.wav")
V7 = os.path.join(ROOT, "outputs", "v7")
FINAL = os.path.join(ROOT, "video", "public", "audio", "final_mix.wav")


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


def rms_db(a):
    mono = a.mean(1)
    return 20 * math.log10(max(float(np.sqrt((mono ** 2).mean())), 1e-9))


def speech_rms_db(a):
    """Same ruler as v6: per-10ms blocks, dB average above -45 dB."""
    mono = a.mean(1)
    n = len(mono) // WIN
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(e, 1e-9))
    loud = db[db > -45.0]
    return float(loud.mean()) if len(loud) else rms_db(a)


def block_db(a):
    mono = a.mean(1)
    n = len(mono) // WIN
    e = np.sqrt((mono[: n * WIN].reshape(n, WIN) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(e, 1e-9))


def duck_curve(narr, speech_db):
    """Per-block gain: ducked under speech, 1.0 in the gaps (same as v6)."""
    db = block_db(narr)
    active = db > (speech_db - 15.0)
    tgt_gain = 10 ** (-DUCK_DB / 20.0)
    att = 1 - math.exp(-0.01 / 0.02)
    rel = 1 - math.exp(-0.01 / 0.40)
    g = np.empty(len(active), np.float32)
    cur = 1.0
    for i in range(len(active)):
        tgt = tgt_gain if active[i] else 1.0
        a = att if tgt < cur else rel
        cur += (tgt - cur) * a
        g[i] = cur
    g_full = np.repeat(g, WIN)
    if len(g_full) < len(narr):
        g_full = np.concatenate([g_full, np.ones(len(narr) - len(g_full), np.float32)])
    return g_full[: len(narr)], active


def main():
    narr = read_wav_f(NARR)
    bed = read_wav_f(BED)
    n = len(narr)
    print(f"narration {len(narr)/SR:.2f}s | bed source {len(bed)/SR:.2f}s")

    # bed must exactly cover the narration timeline (88.0s > 87.45s, so trim)
    bed = bed[:n] if len(bed) >= n else np.concatenate(
        [bed, np.zeros((n - len(bed), 2), np.float32)])

    # short fade-in so the game music does not slam in at frame 0
    fi = int(1.0 * SR)
    bed[:fi] *= np.linspace(0, 1, fi)[:, None]
    # tail fade: the game's clear fanfare peaks at the very end (86.5s+), which
    # would step on the last line. Fade the bed out under the closing line.
    fo_bed = int(2.4 * SR)
    bed[-fo_bed:] *= np.linspace(1, 0, fo_bed)[:, None]

    target = speech_rms_db(narr)
    bed_db = rms_db(bed)
    bed *= 10 ** ((target + BGM_GAP_DB - bed_db) / 20.0)
    bed *= 10 ** (BED_TRIM_DB / 20.0)
    print(f"narration reference {target:.1f} dB | bed {bed_db:.1f} -> "
          f"{target + BGM_GAP_DB + BED_TRIM_DB:.1f} dB RMS "
          f"(incl. {BED_TRIM_DB:+.2f} dB trim)")

    g_full, active = duck_curve(narr, target)
    ducked = bed * g_full[:, None]
    mix = narr + ducked

    peak = float(np.abs(mix).max())
    if peak > 0.97:
        mix *= 0.97 / peak
        print(f"limiting: peak {peak:.3f} -> 0.97")
    fo = int(1.2 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None]

    # level report (same rulers as v6)
    sp = active.astype(bool)
    gaps = ~sp
    print(f"[levels] narration speech {block_db(narr)[sp].mean():6.1f} dB | "
          f"game audio under speech {block_db(ducked)[sp].mean():6.1f} dB | "
          f"in gaps {block_db(ducked)[gaps].mean():6.1f} dB")

    write_wav_f(os.path.join(V7, "ducked_game_bed.wav"), ducked)
    write_wav_f(os.path.join(V7, "final_mix_v7.wav"), mix)
    write_wav_f(FINAL, mix)
    print(f"wrote {FINAL}: {len(mix)/SR:.2f}s, peak {float(np.abs(mix).max()):.3f}")


if __name__ == "__main__":
    main()
