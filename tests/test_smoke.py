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


def test_model_resolution():
    """An explicit --model must win, even when it equals the default."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        here = Path(d)

        # nothing built yet
        assert livecap.resolve_model(None, "fi", here)[0] == "small"
        assert livecap.resolve_model(None, "ru", here)[0] == "small"

        built = here / livecap.FI_MODEL_DIR
        built.mkdir(parents=True)
        (built / "model.bin").write_bytes(b"x")

        name, why = livecap.resolve_model(None, "fi", here)
        assert name == str(built), name
        assert why and "fine-tune" in why

        # the fine-tune is Finnish-only
        assert livecap.resolve_model(None, "ru", here)[0] == "small"
        assert livecap.resolve_model(None, "ja", here)[0] == "small"

        # explicit wins, including the string that happens to be the default
        assert livecap.resolve_model("small", "fi", here) == ("small", None)
        assert livecap.resolve_model("large-v3", "fi", here)[0] == "large-v3"


def _controller():
    import argparse

    args = argparse.Namespace(
        lang="fi", model="small", translate=False, partials=True,
        langs=["fi", "ru", "ja", "es", "pt", "auto"],
        model_choices=["tiny", "base", "small"],
        txt="captions.txt", txt_lines=2)
    bus = livecap.Bus(args)
    bus.publish = lambda msg: bus.history.append(msg)   # no websocket in tests
    return livecap.Controller(args, bus), args


def test_language_codes_validated():
    for good in ("fi", "ru", "ja", "es", "pt", "en", "auto"):
        assert livecap.valid_language(good), good
    for bad in ("klingon", "", None, "e", 42):
        assert not livecap.valid_language(bad), bad


def test_controller_switches_language():
    ctl, args = _controller()
    ctl.handle('{"cmd":"set_lang","value":"ja"}')
    assert args.lang == "ja"
    ctl.handle('{"cmd":"set_lang","value":"ru"}')
    assert args.lang == "ru"


def test_controller_rejects_bad_language_and_bad_json():
    ctl, args = _controller()
    ctl.handle('{"cmd":"set_lang","value":"klingon"}')
    assert args.lang == "fi"
    ctl.handle("not json at all")
    ctl.handle('{"cmd":"nonsense"}')
    assert args.lang == "fi"


def test_model_change_is_queued_for_the_worker_thread():
    """Model loading must not happen on the websocket thread."""
    ctl, args = _controller()
    ctl.handle('{"cmd":"set_model","value":"base"}')
    assert ctl.cmds.get_nowait() == ("model", "base")
    assert args.model == "small", "model must not change until the worker reloads it"

    ctl.handle('{"cmd":"set_model","value":"small"}')   # already current
    assert ctl.cmds.empty()


def test_toggles():
    ctl, args = _controller()
    ctl.handle('{"cmd":"set_translate","value":true}')
    assert args.translate is True
    ctl.handle('{"cmd":"set_partials","value":false}')
    assert args.partials is False


def test_status_payload_is_complete():
    ctl, _ = _controller()
    s = ctl.status()
    for key in ("type", "lang", "model", "loading", "translate",
                "partials", "langs", "models"):
        assert key in s, key
    assert s["type"] == "status"


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
