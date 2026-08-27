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


def test_collapse_repeated_chunks():
    """Consecutive segments repeating the same sentence collapse to one."""
    one = "私はそれを見つけたことがありました"
    assert livecap.collapse_repeats(" ".join([one] * 4)) == one
    assert livecap.collapse_repeats(one) == one


def test_collapse_repeated_unit_inside_a_segment():
    """The loop need not start at the beginning of the string."""
    assert livecap.collapse_repeats("この日の日の日の日") == "この日"
    assert livecap.collapse_repeats("abababab") == "ab"
    assert livecap.collapse_repeats("xabababab") == "xab"


def test_short_words_may_repeat_twice():
    """Doubling is normal speech; a longer run is Whisper looping."""
    assert livecap.collapse_repeats("very very good") == "very very good"
    assert livecap.collapse_repeats("no no no no no") == "no no"


def test_collapse_leaves_normal_text_alone():
    for text in ["今日はとてもいい天気ですね。",
                 "Kyllä se tästä, kyllä se tästä.",
                 "Привет, как дела сегодня?",
                 "the cat sat on the mat"]:
        assert livecap.collapse_repeats(text) == " ".join(text.split()), text


def test_is_repetitive():
    one = "私はそれを見つけたことがありました"
    assert livecap.is_repetitive(" ".join([one] * 4))
    assert not livecap.is_repetitive(one)
    assert not livecap.is_repetitive("Kyllä se tästä, kyllä se tästä.")
    assert not livecap.is_repetitive("hi")


LANGS = ["fi", "ru", "ja", "es", "pt", "en"]


def _det(**kw):
    return livecap.LanguageDetector(LANGS + ["auto"], **kw)


def test_detector_ignores_languages_not_in_use():
    """An outsider may win outright and still not be the answer.

    But only when enough evidence lands inside the set - if the audio is
    overwhelmingly a language not in use, the honest answer is "no idea",
    not the best of the leftovers.
    """
    d = _det()
    # de leads, yet fi holds the majority of the mass that is actually usable
    assert d.observe([("de", 0.3), ("fi", 0.55), ("ja", 0.05)], duration=4) == "fi"

    d2 = _det()
    # here de dominates and almost nothing is in the set: refuse to guess
    assert d2.observe([("de", 0.7), ("fi", 0.2), ("ja", 0.05)], duration=4) is None


def test_detector_folds_close_relatives():
    """Estonian mass belongs to Finnish; Ukrainian to Russian; Galician to Portuguese."""
    d = _det()
    folded, mass = d.fold([("et", 0.5), ("fi", 0.3), ("ru", 0.1)])
    assert folded["fi"] == 0.8, folded
    assert 0.89 < mass < 0.91, mass

    folded, _ = d.fold([("uk", 0.4), ("bg", 0.2), ("ru", 0.1)])
    assert abs(folded["ru"] - 0.7) < 1e-9, folded

    folded, _ = d.fold([("gl", 0.6), ("es", 0.2)])
    assert abs(folded["pt"] - 0.6) < 1e-9 and abs(folded["es"] - 0.2) < 1e-9, folded


def test_detector_splits_do_not_lose_to_an_outsider():
    """fi+et together beat ja, even though ja beats each individually."""
    d = _det()
    assert d.observe([("fi", 0.3), ("et", 0.3), ("ja", 0.35)], duration=4) == "fi"


def test_detector_does_not_flap_on_one_odd_segment():
    d = _det(hold=2)
    for _ in range(4):
        d.observe([("ja", 0.95)], duration=4)
    assert d.current == "ja"
    # a single confident Spanish reading must not switch it
    assert d.observe([("es", 0.95)], duration=4) == "ja"


def test_detector_switches_when_change_is_sustained():
    d = _det(hold=2)
    for _ in range(4):
        d.observe([("ja", 0.95)], duration=4)
    assert d.current == "ja"
    d.observe([("ru", 0.95)], duration=4)
    d.observe([("ru", 0.95)], duration=4)
    for _ in range(3):
        d.observe([("ru", 0.95)], duration=4)
    assert d.current == "ru", d.scores


def test_detector_weights_short_audio_less():
    d = _det()
    for _ in range(3):
        d.observe([("en", 0.9)], duration=5)
    strong = dict(d.scores)
    d2 = _det()
    for _ in range(3):
        d2.observe([("en", 0.9)], duration=0.5)
    assert strong["en"] > d2.scores["en"]


def test_detector_reset_and_confidence():
    d = _det()
    d.observe([("ja", 0.99)], duration=4)
    assert d.current == "ja" and d.confidence() > 0.9
    d.reset()
    assert d.current is None and d.confidence() == 0.0
    assert all(v == 0.0 for v in d.scores.values())


def test_detector_defers_on_an_ambiguous_first_reading():
    """Restricting the set inflates confidence; mass outside it is the tell.

    "en 0.38, ko 0.25, nn 0.10" looks like a commanding en once ko and nn are
    dropped, but only 0.38 of the mass was ever inside the allowed set.
    """
    d = _det()
    assert d.observe([("en", 0.38), ("ko", 0.25), ("nn", 0.10)], duration=2) is None
    assert d.current is None
    # a genuinely confident reading is adopted immediately afterwards
    assert d.observe([("fi", 0.96), ("nn", 0.02), ("en", 0.01)], duration=4) == "fi"


def test_detector_adopts_a_confident_first_reading():
    d = _det()
    assert d.observe([("ja", 0.93), ("zh", 0.04)], duration=4) == "ja"


def test_detector_low_mass_never_locks_in():
    d = _det()
    for _ in range(5):
        d.observe([("de", 0.6), ("nl", 0.3), ("en", 0.05)], duration=4)
    assert d.current is None, d.scores


def test_detector_handles_empty_and_unusable_input():
    d = _det()
    assert d.observe([], duration=4) is None
    assert d.observe([("de", 0.9), ("zh", 0.1)], duration=4) is None
    d.observe([("fi", 0.9)], duration=4)
    assert d.observe([("de", 1.0)], duration=4) == "fi"   # keeps the last good one


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
