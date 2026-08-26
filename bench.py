#!/usr/bin/env python3
"""
bench.py - how fast can this machine transcribe? Picks your --model for you.

    python bench.py                      # synthetic audio, default model set
    python bench.py --models small,medium,large-v3-turbo
    python bench.py --wav selftest.wav   # far more accurate: use real speech

What matters for live captions is the wall-clock time to transcribe one
segment, because that is the delay between someone finishing a sentence and
the caption appearing. Whisper's encoder always runs over a padded 30s window,
so that cost is roughly constant no matter how short the segment is.
"""

import argparse
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGET_SR = 16000


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != TARGET_SR or w.getnchannels() != 1:
            raise SystemExit("%s must be 16 kHz mono (selftest.wav always is)." % path)
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def synth(seconds):
    """Speech-ish: a few formants, amplitude-modulated at a syllable rate."""
    t = np.arange(int(TARGET_SR * seconds)) / TARGET_SR
    rng = np.random.default_rng(0)
    sig = np.zeros_like(t)
    for f in (140, 420, 900, 1800, 2600):
        sig += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / f ** 0.35
    syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t)
    sig = sig * syllables + 0.01 * rng.standard_normal(len(t))
    return (0.3 * sig / np.max(np.abs(sig))).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="tiny,base,small,large-v3-turbo")
    p.add_argument("--lang", default="fi")
    p.add_argument("--compute", default="int8")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--wav", default=None, help="16 kHz mono wav of real speech")
    args = p.parse_args()

    import os
    from faster_whisper import WhisperModel

    threads = args.threads or max(2, (os.cpu_count() or 8) - 2)
    if args.wav:
        audio = load_wav(Path(args.wav) if Path(args.wav).is_absolute() else HERE / args.wav)
        source = args.wav
    else:
        audio = synth(args.seconds)
        source = "synthetic (install-free, but optimistic on decode time)"
    dur = len(audio) / TARGET_SR

    print("\naudio: %s  (%.1fs)   threads: %d   compute: %s\n"
          % (source, dur, threads, args.compute))
    print("%-20s %9s %9s %8s   %s" % ("model", "load", "per-seg", "RTF", "verdict"))
    print("-" * 72)

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            t0 = time.time()
            model = WhisperModel(name, device="cpu", compute_type=args.compute,
                                 cpu_threads=threads, num_workers=1)
            load = time.time() - t0

            times = []
            for i in range(args.runs):
                t0 = time.time()
                segs, _ = model.transcribe(
                    audio, language=None if args.lang == "auto" else args.lang,
                    beam_size=5, temperature=0.0, condition_on_previous_text=False,
                    vad_filter=False, without_timestamps=True)
                list(segs)                      # generator: force the work
                times.append(time.time() - t0)
            best = min(times[1:] or times)      # first run includes warm-up
            rtf = best / dur

            if rtf < 0.35:
                verdict = "excellent - use this"
            elif rtf < 0.6:
                verdict = "fine for live"
            elif rtf < 1.0:
                verdict = "tight, captions will lag"
            else:
                verdict = "too slow"
            print("%-20s %8.1fs %8.2fs %8.2f   %s" % (name, load, best, rtf, verdict))
            del model
        except Exception as e:
            print("%-20s %s" % (name, repr(e)[:60]))

    print("\nper-seg = delay between end of a sentence and its caption.")
    print("Anything under ~0.6 RTF keeps up with continuous speech.\n")


if __name__ == "__main__":
    main()
