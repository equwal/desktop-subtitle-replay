"""Dependency-free smoke tests. Run with:  python tests\\test_smoke.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import livecap


def test_hallucination_filter():
    """Stock Whisper-on-silence phrases are dropped, real speech is kept."""
    for bad in ("Tekstitys: YLE 2021", "Kiitos kun katsoit!", "Kiitos.",
                "Thanks for watching", "Subtitles by someone", "   "):
        assert livecap.looks_hallucinated(bad), bad
    for good in ("Moi, mita kuuluu tanaan?",
                 "Nyt ollaan taas sen verran syrjaisilla seuduilla",
                 "Kiitos kun tulit mukaan, puhutaan seuraavaksi saasta ja "
                 "siita mita ensi viikolla tapahtuu"):
        assert not livecap.looks_hallucinated(good), good


def test_resample_to_whisper_rate():
    """48 kHz in, 16 kHz out, still in range, tone preserved."""
    sr = 48000
    t = np.arange(sr) / sr
    sig = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    out = livecap.to_whisper(sig, sr)

    assert out.dtype == np.float32, out.dtype
    assert abs(len(out) - livecap.TARGET_SR) <= 2, len(out)
    assert np.max(np.abs(out)) <= 1.0

    spec = np.abs(np.fft.rfft(out))
    freq = np.fft.rfftfreq(len(out), 1 / livecap.TARGET_SR)[int(np.argmax(spec))]
    assert 430 < freq < 450, freq


def test_quiet_audio_is_gained_up():
    """Very quiet input is normalised toward a level Whisper can use."""
    quiet = (0.01 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    out = livecap.to_whisper(quiet, livecap.TARGET_SR)
    assert np.max(np.abs(out)) > 0.3, np.max(np.abs(out))


def test_silence_does_not_divide_by_zero():
    out = livecap.to_whisper(np.zeros(16000, dtype=np.float32), livecap.TARGET_SR)
    assert np.all(out == 0)
    assert len(out) == 16000


def test_captions_file_keeps_last_n_lines(tmp_path=None):
    """captions.txt is a rolling window; captions.log keeps everything."""
    import argparse
    import os
    import tempfile

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            livecap.HERE = Path(d)
            bus = livecap.Bus(argparse.Namespace(
                txt="captions.txt", txt_lines=2, translate=False))
            for line in ("eka", "toka", "kolmas"):
                bus.publish({"type": "final", "text": line, "tr": "", "ts": 0})

            assert (Path(d) / "captions.txt").read_text(encoding="utf-8").split() \
                == ["toka", "kolmas"]
            log = (Path(d) / "captions.log").read_text(encoding="utf-8")
            assert log.count("\n") == 3 and "eka" in log
        finally:
            os.chdir(cwd)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("  PASS  %s" % fn.__name__)
        except Exception as e:
            failed += 1
            print("  FAIL  %s: %r" % (fn.__name__, e))
    print("\n%d passed, %d failed" % (len(tests) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
