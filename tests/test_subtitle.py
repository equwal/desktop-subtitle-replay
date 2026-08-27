"""Tests for the offline subtitler. Run with:  python tests\\test_subtitle.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subtitle


class W:
    """Minimal stand-in for a faster-whisper word."""

    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class Seg:
    def __init__(self, words, text=None, start=0.0, end=0.0):
        self.words = words
        self.text = text if text is not None else "".join(w.word for w in words)
        self.start, self.end = start, end


def words(spec, t0=0.0, step=0.4):
    out, t = [], t0
    for token in spec.split(" "):
        out.append(W(t, t + step, token + " "))
        t += step
    return out


def test_timestamp_format():
    assert subtitle.ts(0) == "00:00:00,000"
    assert subtitle.ts(1.5) == "00:00:01,500"
    assert subtitle.ts(3661.25) == "01:01:01,250"
    assert subtitle.ts(-5) == "00:00:00,000"
    assert subtitle.ts(1.5, ".") == "00:00:01.500"


def test_wrap_balances_two_lines():
    out = subtitle.wrap("aaa bbb ccc ddd", 8)
    assert out == "aaa bbb\nccc ddd", repr(out)


def test_wrap_leaves_short_text_alone():
    assert subtitle.wrap("short", 42) == "short"


def test_wrap_falls_back_when_nothing_fits():
    """A long sentence with no split under the limit still gets two lines."""
    text = ("Ollaan taas sen verran syrjaisilla seuduilla ja "
            "harvakseltaan kuljetuilla seuduilla.")
    out = subtitle.wrap(text, 42)
    lines = out.split("\n")
    assert len(lines) == 2, out
    assert max(len(x) for x in lines) < len(text), out


def test_wrap_single_word_cannot_split():
    assert "\n" not in subtitle.wrap("Rindfleischetikettierungsgesetz", 5)


def test_wrap_japanese_has_no_spaces_to_split_on():
    text = "今日はとてもいい天気ですね。散歩に行きましょうか。"
    out = subtitle.wrap(text, 16)
    lines = out.split("\n")
    assert len(lines) == 2, out
    assert max(len(x) for x in lines) <= 16, out
    assert "".join(lines) == text


def test_wrap_japanese_prefers_breaking_after_punctuation():
    text = "今日はいい天気。散歩に行こう。"
    out = subtitle.wrap(text, 10)
    assert out.split("\n")[0].endswith("。"), out


def test_wrap_japanese_never_starts_a_line_with_closing_marks():
    for text in ["これはテストです、そしてこれも試験です。",
                 "彼は「そうだね」と言ったのでした。",
                 "ちょっとまってっていったよね。"]:
        out = subtitle.wrap(text, 10)
        for line in out.split("\n")[1:]:
            assert line[0] not in subtitle.NO_LINE_START, (text, out)


def test_looks_cjk():
    assert subtitle.looks_cjk("今日は")
    assert subtitle.looks_cjk("テスト")
    assert not subtitle.looks_cjk("Kyllä se tästä")
    assert not subtitle.looks_cjk("Привет")


def test_sentences_split_on_terminators():
    ws = words("Yksi kaksi.") + words("Kolme nelja.", t0=2.0)
    groups = subtitle.group_sentences(ws, max_dur=6.0, max_gap=0.8)
    assert len(groups) == 2, [subtitle.text_of(g) for g in groups]
    assert subtitle.text_of(groups[0]) == "Yksi kaksi."


def test_sentences_split_on_long_gap():
    ws = words("yksi kaksi") + words("kolme nelja", t0=9.0)
    groups = subtitle.group_sentences(ws, max_dur=6.0, max_gap=0.8)
    assert len(groups) == 2


def test_cue_never_starts_mid_sentence_orphan():
    """A trailing runt is folded back rather than shown alone."""
    ws = words("aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk")
    cues = subtitle.build_cues([Seg(ws)], max_chars=30, max_dur=99, max_gap=9)
    assert cues
    for _, _, text in cues:
        assert len(text) >= 16 or len(cues) == 1, cues


def test_segment_without_word_timings_still_produces_a_cue():
    seg = Seg([], text="Ei sanatason aikaleimoja.", start=1.0, end=3.0)
    cues = subtitle.build_cues([seg], 84, 6.0, 0.8)
    assert len(cues) == 1
    assert cues[0][2] == "Ei sanatason aikaleimoja."
    assert cues[0][0] == 1.0 and cues[0][1] == 3.0


def test_srt_roundtrip(tmp=None):
    import tempfile

    cues = [(0.0, 1.5, "Ensimmainen."), (2.0, 3.25, "Toinen rivi tassa.")]
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.srt"
        subtitle.write_srt(cues, out, 42)
        text = out.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,500\nEnsimmainen." in text, text
    assert "2\n00:00:02,000 --> 00:00:03,250" in text, text
    assert text.endswith("\n\n")


def test_vtt_has_header_and_dot_timestamps():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.vtt"
        subtitle.write_vtt([(0.0, 1.0, "Moi")], out, 42)
        text = out.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in text


def test_mine_page_embeds_cues_and_escapes_nothing_odd():
    import json
    import tempfile

    cues = [(0.0, 1.0, "今日はいい天気ですね。"), (1.0, 2.0, "Kyllä se tästä.")]
    with tempfile.TemporaryDirectory() as d:
        video = Path(d) / "clip.mp4"
        video.write_bytes(b"")
        page = subtitle.write_mine_page(video, cues, "ja")
        html = page.read_text(encoding="utf-8")
    assert page.name == "clip.html"
    assert 'src="clip.mp4"' in html
    # Non-Latin text must survive verbatim for Yomitan to scan it.
    assert "今日はいい天気ですね。" in html
    assert "Kyllä se tästä." in html
    assert "__CUES__" not in html and "__VIDEO__" not in html
    start = html.index("const CUES = ") + len("const CUES = ")
    data = json.loads(html[start:html.index("\n", start)].rstrip(";"))
    assert len(data) == 2 and data[0]["text"] == "今日はいい天気ですね。"


def test_esc_strips_tabs_and_newlines():
    assert subtitle.esc("a\tb\nc ") == "a b c"


def test_obs_folder_detection_handles_bom_and_output_mode():
    """OBS writes its ini files with a UTF-8 BOM, which trips configparser."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "obs-studio"
        prof = root / "basic" / "profiles" / "Untitled"
        prof.mkdir(parents=True)
        recdir = Path(d) / "recordings"
        recdir.mkdir()
        simpledir = Path(d) / "simple"
        simpledir.mkdir()

        (root / "user.ini").write_text(
            "﻿[Basic]\nProfile=Untitled\nProfileDir=Untitled\n", encoding="utf-8")
        (prof / "basic.ini").write_text(
            "﻿[Output]\nMode=Advanced\n\n"
            "[SimpleOutput]\nFilePath=%s\n\n"
            "[AdvOut]\nRecFilePath=%s\n"
            % (str(simpledir).replace("\\", "\\\\"),
               str(recdir).replace("\\", "\\\\")),
            encoding="utf-8")

        old = os.environ.get("APPDATA")
        os.environ["APPDATA"] = d
        try:
            found = subtitle.obs_recording_folder()
        finally:
            if old is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old

    assert found is not None, "BOM or mode parsing failed"
    assert Path(found) == recdir, (found, recdir)


def test_obs_folder_detection_simple_mode():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "obs-studio"
        prof = root / "basic" / "profiles" / "P"
        prof.mkdir(parents=True)
        simpledir = Path(d) / "simple"
        simpledir.mkdir()
        (root / "user.ini").write_text("﻿[Basic]\nProfileDir=P\n", encoding="utf-8")
        (prof / "basic.ini").write_text(
            "﻿[Output]\nMode=Simple\n\n[SimpleOutput]\nFilePath=%s\n"
            % str(simpledir).replace("\\", "\\\\"), encoding="utf-8")

        old = os.environ.get("APPDATA")
        os.environ["APPDATA"] = d
        try:
            found = subtitle.obs_recording_folder()
        finally:
            if old is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old
    assert Path(found) == simpledir, found


def test_range_requests_are_served():
    """Without 206 support a browser will not seek in a <video> at all."""
    import http.client
    import tempfile

    payload = bytes(range(256)) * 8          # 2048 bytes
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "clip.mp4").write_bytes(payload)
        srv = subtitle.serve(Path(d), 0)     # port 0 = pick a free one
        port = srv.server_address[1]
        try:
            def req(headers):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                c.request("GET", "/clip.mp4", headers=headers)
                r = c.getresponse()
                body = r.read()
                c.close()
                return r, body

            r, body = req({})
            assert r.status == 200, r.status
            assert r.getheader("Accept-Ranges") == "bytes"
            assert body == payload

            r, body = req({"Range": "bytes=10-19"})
            assert r.status == 206, r.status
            assert r.getheader("Content-Range") == "bytes 10-19/2048"
            assert body == payload[10:20], body

            r, body = req({"Range": "bytes=2040-"})
            assert r.status == 206
            assert body == payload[2040:]

            r, body = req({"Range": "bytes=-8"})
            assert r.status == 206
            assert body == payload[-8:]

            r, _ = req({"Range": "bytes=99999-"})
            assert r.status == 416, r.status
        finally:
            srv.shutdown()
            srv.server_close()


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
