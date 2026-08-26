#!/usr/bin/env python3
"""
subtitle.py - subtitle recordings and replay-buffer clips.

Latency does not matter here, so this uses a much larger model than the live
captioner can afford and produces far better text, especially for Finnish,
Russian and Japanese.

    python subtitle.py clip.mp4
    python subtitle.py *.mkv --lang ru
    python subtitle.py --watch "C:\\Users\\me\\Videos"      # auto-subtitle new clips
    python subtitle.py clip.mp4 --format vtt --translate

Writes clip.srt next to clip.mp4. OBS, VLC, YouTube and Premiere all read it.
No ffmpeg needed - audio is decoded through PyAV, which ships with
faster-whisper.
"""

import argparse
import os
import re
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".flv", ".webm", ".avi", ".ts",
             ".m4a", ".mp3", ".wav", ".opus", ".flac", ".ogg"}

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(*a):
    print("[" + time.strftime("%H:%M:%S") + "]", *a, flush=True)


# ------------------------------------------------------------------ cues ----

def ts(seconds, sep=","):
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d%s%03d" % (h, m, s, sep, ms)


def wrap(text, width):
    """Split into at most two balanced lines, the way subtitles are normally set."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    words = text.split()
    if len(words) < 2:
        return text

    fits, over = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        longest, balance = max(len(a), len(b)), abs(len(a) - len(b))
        if longest <= width:
            if fits is None or balance < fits[0]:
                fits = (balance, a, b)
        # Fallback for text with no split that fits: overflow as little as
        # possible rather than emitting one very long line.
        if over is None or (longest, balance) < (over[0], over[1]):
            over = (longest, balance, a, b)

    if fits:
        return fits[1] + "\n" + fits[2]
    return over[2] + "\n" + over[3]


SENT_END = (".", "!", "?", "…", "。", "！", "？", "؟")


class _Word:
    """Stand-in when Whisper returns a segment without word timings."""

    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


def collect_words(segments):
    words = []
    for seg in segments:
        ws = list(getattr(seg, "words", None) or [])
        if ws:
            words.extend(ws)
        elif seg.text.strip():
            words.append(_Word(seg.start, seg.end, seg.text.strip() + " "))
    return words


def group_sentences(words, max_dur, max_gap):
    """Split on sentence endings first - this is the boundary that matters.

    Long pauses and a hard duration cap act only as fallbacks, so a cue never
    starts mid-sentence just because a character budget ran out.
    """
    out, cur = [], []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            dur = w.end - cur[0].start
            if gap > max_gap or dur > max_dur * 2.5:
                out.append(cur)
                cur = []
        cur.append(w)
        if w.word.strip().endswith(SENT_END):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [g for g in out if "".join(x.word for x in g).strip()]


def text_of(group):
    return " ".join("".join(w.word for w in group).split())


def build_cues(segments, max_chars, max_dur, max_gap):
    """Sentences first, then subdivide any that are too long to display."""
    cues = []
    for group in group_sentences(collect_words(segments), max_dur, max_gap):
        chunks, chunk = [], []
        for w in group:
            if chunk:
                chars = sum(len(x.word) for x in chunk) + len(w.word)
                dur = w.end - chunk[0].start
                if chars > max_chars or dur > max_dur:
                    chunks.append(chunk)
                    chunk = []
            chunk.append(w)
        if chunk:
            chunks.append(chunk)

        # Greedy filling can strand a word or two on the last line; fold a
        # runt back into its predecessor rather than flashing it alone.
        if len(chunks) > 1 and len("".join(w.word for w in chunks[-1]).strip()) < 16:
            chunks[-2].extend(chunks.pop())

        for c in chunks:
            cues.append((c[0].start, c[-1].end, text_of(c)))
    return cues


def write_srt(cues, path, width):
    with open(path, "w", encoding="utf-8") as fh:
        for i, (start, end, text) in enumerate(cues, 1):
            fh.write("%d\n%s --> %s\n%s\n\n"
                     % (i, ts(start), ts(end), wrap(text, width)))


def write_vtt(cues, path, width):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("WEBVTT\n\n")
        for start, end, text in cues:
            fh.write("%s --> %s\n%s\n\n"
                     % (ts(start, "."), ts(end, "."), wrap(text, width)))


def write_txt(cues, path, width):
    with open(path, "w", encoding="utf-8") as fh:
        for _, _, text in cues:
            fh.write(text + "\n")


WRITERS = {"srt": write_srt, "vtt": write_vtt, "txt": write_txt}


# ------------------------------------------------------------------ anki ----

def anki_media_dir():
    """Anki's collection.media for the default profile, if we can find it."""
    base = Path(os.environ.get("APPDATA", "")) / "Anki2"
    if not base.is_dir():
        return None
    for profile in sorted(base.iterdir()):
        media = profile / "collection.media"
        if media.is_dir():
            return media
    return None


def slice_audio(audio, sr, start, end, pad):
    lo = max(0, int((start - pad) * sr))
    hi = min(len(audio), int((end + pad) * sr))
    return audio[lo:hi]


def write_audio_clip(samples, sr, path):
    """mp3 when PyAV has an encoder for it, otherwise wav."""
    import numpy as np

    pcm = np.clip(samples, -1.0, 1.0)
    if path.suffix == ".mp3":
        try:
            import av

            with av.open(str(path), "w") as container:
                stream = container.add_stream("mp3", rate=sr)
                stream.layout = "mono"
                frame = av.AudioFrame.from_ndarray(
                    (pcm * 32767).astype(np.int16).reshape(1, -1),
                    format="s16", layout="mono")
                frame.rate = sr
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            return path
        except Exception:
            path = path.with_suffix(".wav")

    import wave

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    return path


def grab_frames(video, times, out_paths, width):
    """Best-effort screenshots. Returns the paths actually written."""
    try:
        import av
    except ImportError:
        return {}
    written = {}
    try:
        with av.open(str(video)) as container:
            if not container.streams.video:
                return {}
            vs = container.streams.video[0]
            vs.thread_type = "AUTO"
            for i, (t, dest) in enumerate(zip(times, out_paths)):
                try:
                    container.seek(int(t / vs.time_base), stream=vs)
                    for frame in container.decode(vs):
                        if frame.time is None or frame.time + 0.001 < t:
                            continue
                        img = frame.to_image()
                        if width and img.width > width:
                            img = img.resize((width, max(1, round(img.height * width / img.width))))
                        img.save(str(dest), quality=82)
                        written[i] = dest
                        break
                except Exception:
                    continue
    except Exception:
        return written
    return written


def esc(text):
    return text.replace("\t", " ").replace("\n", " ").strip()


# ----------------------------------------------------------- mining page ----

MINE_PAGE = """<!doctype html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { --bg:#0e1116; --panel:#161a21; --line:#252b36; --fg:#edf1f7;
          --dim:#8b95a7; --accent:#7dd3fc; --size:26px; }
  * { box-sizing:border-box; }
  html,body { height:100%; margin:0; }
  body { background:var(--bg); color:var(--fg); display:flex; flex-direction:column;
         font-family:"Inter","Segoe UI","Yu Gothic UI","Meiryo","Noto Sans CJK JP",
                     "Noto Sans",system-ui,sans-serif; }
  header { display:flex; gap:10px; align-items:center; flex-wrap:wrap; padding:10px 16px;
           background:var(--panel); border-bottom:1px solid var(--line); font-size:13px; }
  header .grow { flex:1; }
  button { font:inherit; font-size:13px; color:var(--fg); cursor:pointer; background:#1f2531;
           border:1px solid var(--line); border-radius:6px; padding:6px 11px; }
  button:hover { background:#29313f; }
  button.on { background:var(--accent); border-color:var(--accent); color:#06202c; }
  main { flex:1; display:flex; min-height:0; flex-wrap:wrap; }
  #left { flex:1 1 460px; min-width:320px; display:flex; flex-direction:column;
          border-right:1px solid var(--line); }
  video { width:100%; background:#000; max-height:52vh; }
  #cur { padding:16px 20px; font-size:var(--size); line-height:1.7; min-height:3em;
         user-select:text; border-top:1px solid var(--line); }
  #list { flex:1 1 380px; min-width:300px; overflow-y:auto; padding:14px 18px 30vh; }
  .cue { padding:8px 11px; border-radius:6px; border-left:3px solid transparent;
         font-size:calc(var(--size)*.82); line-height:1.7; user-select:text; margin-bottom:8px; }
  .cue::before { content:attr(data-time); display:block; font-size:11px; color:var(--dim);
                 user-select:none; margin-bottom:2px; font-variant-numeric:tabular-nums; }
  .cue:hover { background:#151a22; }
  .cue.active { background:#1a222e; border-left-color:var(--accent); }
  .jump { float:right; font-size:11px; color:var(--dim); cursor:pointer; user-select:none; }
</style>
</head>
<body>
<header>
  <b>__TITLE__</b>
  <span class="grow"></span>
  <button id="smaller">A-</button>
  <button id="bigger">A+</button>
  <button id="follow" class="on">Follow</button>
  <button id="loop">Loop cue</button>
  <button id="copy">Copy all</button>
</header>
<main>
  <div id="left">
    <video id="v" src="__VIDEO__" controls preload="metadata"></video>
    <div id="cur"></div>
  </div>
  <div id="list"></div>
</main>
<script>
const CUES = __CUES__;
const v = document.getElementById("v");
const list = document.getElementById("list");
const cur = document.getElementById("cur");
let follow = true, looping = false, active = -1;

function fmt(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
}

CUES.forEach((c, i) => {
  const d = document.createElement("div");
  d.className = "cue";
  d.dataset.time = fmt(c.start);
  const j = document.createElement("span");
  j.className = "jump";
  j.textContent = "play";
  j.onclick = (e) => { e.stopPropagation(); v.currentTime = c.start; v.play(); };
  d.appendChild(j);
  d.appendChild(document.createTextNode(c.text));
  d.onclick = () => { v.currentTime = c.start; };
  list.appendChild(d);
});

function setActive(i) {
  if (i === active) return;
  const prev = list.children[active];
  if (prev) prev.classList.remove("active");
  active = i;
  const el = list.children[i];
  cur.textContent = "";
  if (!el) return;
  el.classList.add("active");
  cur.appendChild(document.createTextNode(CUES[i].text));
  if (follow) el.scrollIntoView({ block: "center", behavior: "smooth" });
}

function syncNow() {
  const t = v.currentTime;
  if (looping && active >= 0 && !v.paused) {
    const c = CUES[active];
    if (t > c.end + 0.05) { v.currentTime = c.start; return; }
  }
  let i = -1;
  for (let k = 0; k < CUES.length; k++) {
    if (t >= CUES[k].start - 0.05 && t <= CUES[k].end + 0.35) { i = k; break; }
  }
  // While scrubbing between cues, keep showing the one just passed.
  if (i < 0) {
    for (let k = CUES.length - 1; k >= 0; k--) {
      if (t >= CUES[k].start) { i = k; break; }
    }
  }
  if (i >= 0) setActive(i);
}

// timeupdate only fires during playback; seeked covers scrubbing while paused,
// which is most of what mining actually involves.
["timeupdate", "seeked", "loadedmetadata", "play"].forEach(
  (ev) => v.addEventListener(ev, syncNow));

document.getElementById("bigger").onclick = () => bump(3);
document.getElementById("smaller").onclick = () => bump(-3);
function bump(d) {
  const s = parseInt(getComputedStyle(document.documentElement)
        .getPropertyValue("--size"), 10) + d;
  document.documentElement.style.setProperty("--size",
        Math.max(14, Math.min(60, s)) + "px");
}
document.getElementById("follow").onclick = (e) => {
  follow = !follow; e.target.className = follow ? "on" : "";
};
document.getElementById("loop").onclick = (e) => {
  looping = !looping; e.target.className = looping ? "on" : "";
};
document.getElementById("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText(CUES.map((c) => c.text).join("\\n"));
    const b = document.getElementById("copy");
    b.textContent = "Copied"; setTimeout(() => b.textContent = "Copy all", 1200);
  } catch (e) {}
};
document.addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  if (e.key === "ArrowLeft" && active > 0) v.currentTime = CUES[active - 1].start;
  if (e.key === "ArrowRight" && active < CUES.length - 1) v.currentTime = CUES[active + 1].start;
});
</script>
</body>
</html>
"""


def write_mine_page(path, cues, lang):
    """A self-contained page: the clip plus selectable, synced subtitles.

    Yomitan mines from real DOM text, so the value is in the subtitles being
    hoverable HTML rather than pixels burned into the video.
    """
    import json

    data = [{"start": round(s, 3), "end": round(e, 3), "text": t} for s, e, t in cues]
    html = (MINE_PAGE
            .replace("__CUES__", json.dumps(data, ensure_ascii=False))
            .replace("__VIDEO__", path.name.replace('"', "%22"))
            .replace("__TITLE__", path.stem.replace("<", "&lt;"))
            .replace("__LANG__", "ja" if lang == "ja" else (lang or "en")))
    out = path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def export_anki(path, cues, translations, args, log=log):
    """One row per sentence: text, translation, audio, screenshot."""
    from faster_whisper.audio import decode_audio

    media = Path(args.anki_media) if args.anki_media else path.parent / (path.stem + "_media")
    media.mkdir(parents=True, exist_ok=True)

    sr = 24000
    try:
        audio = decode_audio(str(path), sampling_rate=sr)
    except Exception as e:
        log("cannot re-decode audio for cards: %s" % e)
        return None

    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in path.stem)[:48]
    audio_names, image_names = [], []

    for i, (start, end, _) in enumerate(cues, 1):
        clip = slice_audio(audio, sr, start, end, args.anki_pad)
        dest = write_audio_clip(clip, sr, media / ("%s_%04d.mp3" % (stem, i)))
        audio_names.append(dest.name)

    if args.anki_images:
        mids = [(s + e) / 2 for s, e, _ in cues]
        dests = [media / ("%s_%04d.jpg" % (stem, i)) for i in range(1, len(cues) + 1)]
        got = grab_frames(path, mids, dests, args.anki_image_width)
        image_names = [dests[i].name if i in got else "" for i in range(len(cues))]
        if not got:
            log("no screenshots (PyAV/Pillow could not decode video frames)")
    else:
        image_names = [""] * len(cues)

    tsv = path.with_suffix(".anki.tsv")
    with open(tsv, "w", encoding="utf-8", newline="") as fh:
        for i, (start, end, text) in enumerate(cues):
            row = [
                esc(text),
                esc(translations[i]) if translations else "",
                "[sound:%s]" % audio_names[i],
                '<img src="%s">' % image_names[i] if image_names[i] else "",
                esc(path.name),
                ts(start),
            ]
            fh.write("\t".join(row) + "\n")

    log("%d cards -> %s" % (len(cues), tsv.name))
    log("   media -> %s" % media)
    if not args.anki_media:
        target = anki_media_dir()
        if target:
            log("   copy the media files into: %s" % target)
        else:
            log("   copy the media files into your Anki collection.media folder")
    return tsv


# ------------------------------------------------------------ processing ----

class Subtitler:
    def __init__(self, args):
        from faster_whisper import WhisperModel

        self.args = args
        log("loading %r on %s/%s ..." % (args.model, args.device, args.compute))
        t0 = time.time()
        self.model = WhisperModel(args.model, device=args.device,
                                  compute_type=args.compute,
                                  cpu_threads=args.threads, num_workers=1)
        log("model ready in %.1fs" % (time.time() - t0))

    def run(self, path):
        from faster_whisper.audio import decode_audio

        a = self.args
        out = path.with_suffix("." + a.format)
        if out.exists() and not a.overwrite:
            log("skip (exists): %s" % out.name)
            return out

        try:
            audio = decode_audio(str(path), sampling_rate=16000)
        except Exception as e:
            log("cannot decode %s: %s" % (path.name, e))
            return None
        dur = len(audio) / 16000
        if dur < 0.2:
            log("skip (no audio): %s" % path.name)
            return None

        t0 = time.time()
        segments, info = self.model.transcribe(
            audio,
            language=None if a.lang == "auto" else a.lang,
            task="translate" if a.translate else "transcribe",
            beam_size=a.beam,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=True,
            vad_filter=True,
            word_timestamps=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        segments = list(segments)
        took = time.time() - t0

        if a.anki:
            # Cards want whole sentences; subtitles want display-sized chunks.
            groups = group_sentences(collect_words(segments), a.max_dur, a.max_gap)
            card_cues = [(g[0].start, g[-1].end, text_of(g)) for g in groups]
        else:
            card_cues = []

        cues = build_cues(segments, a.max_chars, a.max_dur, a.max_gap)
        if not cues:
            log("no speech found in %s" % path.name)
            return None

        WRITERS[a.format](cues, out, a.width)
        log("%s  ->  %s  (%d cues, %s, %.0fs audio in %.0fs)"
            % (path.name, out.name, len(cues),
               getattr(info, "language", a.lang), dur, took))

        if a.mine:
            page = write_mine_page(path, cues, getattr(info, "language", a.lang))
            log("mining page  ->  %s" % page.name)

        if a.anki:
            translations = self.translate_cues(audio, card_cues) if a.anki_translate else None
            export_anki(path, card_cues, translations, a)
        return out

    def translate_cues(self, audio, cues):
        """English for the back of each card, translated per sentence."""
        out = []
        log("translating %d sentences for card backs..." % len(cues))
        for start, end, _ in cues:
            lo, hi = max(0, int(start * 16000)), min(len(audio), int(end * 16000))
            clip = audio[lo:hi]
            if len(clip) < 1600:
                out.append("")
                continue
            try:
                segs, _ = self.model.transcribe(
                    clip, language=None if self.args.lang == "auto" else self.args.lang,
                    task="translate", beam_size=self.args.beam, temperature=0.0,
                    condition_on_previous_text=False, vad_filter=False,
                    without_timestamps=True)
                out.append(" ".join(s.text.strip() for s in segs).strip())
            except Exception:
                out.append("")
        return out


class _Limited:
    """Feeds copyfile only the bytes belonging to the requested range."""

    def __init__(self, fh, remaining):
        self.fh, self.remaining = fh, remaining

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        if size is None or size < 0:
            size = self.remaining
        data = self.fh.read(min(size, self.remaining))
        self.remaining -= len(data)
        return data

    def close(self):
        self.fh.close()


class RangeHandler(SimpleHTTPRequestHandler):
    """Static files with HTTP Range support.

    Browsers refuse to seek in a <video> unless the server answers range
    requests, and Python's stock handler does not - so scrubbing a clip, which
    is most of what mining involves, would silently not work.
    """

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        SimpleHTTPRequestHandler.end_headers(self)

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng:
            return SimpleHTTPRequestHandler.send_head(self)

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return SimpleHTTPRequestHandler.send_head(self)
        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(fh.fileno()).st_size
        m = re.match(r"bytes=(\d*)-(\d*)\s*$", rng.strip())
        if not m or (not m.group(1) and not m.group(2)):
            fh.close()
            self.send_error(400, "Malformed Range")
            return None

        if not m.group(1):                       # bytes=-N  (last N bytes)
            start, end = max(0, size - int(m.group(2))), size - 1
        else:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)

        if start >= size or start > end:
            fh.close()
            self.send_response(416)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        fh.seek(start)
        return _Limited(fh, end - start + 1)


def serve(folder, port):
    from functools import partial

    handler = partial(RangeHandler, directory=str(folder))
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log("serving %s at http://127.0.0.1:%d/  (video seeking enabled)" % (folder, port))
    pages = sorted(p.name for p in Path(folder).glob("*.html"))
    for name in pages[-10:]:
        log("   http://127.0.0.1:%d/%s" % (port, name))
    return srv


def stable(path, checks=3, delay=1.0):
    """Wait until a file stops growing, so we do not read a half-written clip."""
    last = -1
    for _ in range(120):
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size == last:
            checks -= 1
            if checks <= 0:
                return size > 0
        else:
            checks = 3
            last = size
        time.sleep(delay)
    return False


def watch(sub, folder, args):
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit("not a folder: %s" % folder)
    seen = {p.resolve() for p in folder.iterdir()
            if p.suffix.lower() in VIDEO_EXT}
    log("watching %s for new clips (%d already present) - Ctrl+C to stop"
        % (folder, len(seen)))
    while True:
        try:
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in VIDEO_EXT:
                    continue
                r = p.resolve()
                if r in seen:
                    continue
                seen.add(r)
                log("new clip: %s" % p.name)
                if stable(p):
                    sub.run(p)
                else:
                    log("gave up waiting for %s to finish writing" % p.name)
            time.sleep(args.poll)
        except KeyboardInterrupt:
            print()
            log("stopped")
            return


def main():
    p = argparse.ArgumentParser(
        description="Subtitle recordings and replay clips",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("files", nargs="*", help="video/audio files to subtitle")
    p.add_argument("--watch", metavar="FOLDER",
                   help="watch a folder and subtitle clips as they appear")
    p.add_argument("--poll", type=float, default=3.0, help="watch interval (s)")

    p.add_argument("--model", default="large-v3-turbo",
                   help="quality matters more than speed here")
    p.add_argument("--lang", default="fi", help="language code, or 'auto'")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--compute", default="int8")
    p.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--translate", action="store_true",
                   help="write English subtitles instead of the original language")

    p.add_argument("--format", default="srt", choices=sorted(WRITERS))
    p.add_argument("--width", type=int, default=42, help="max characters per line")
    p.add_argument("--max-chars", type=int, default=84, help="max characters per cue")
    p.add_argument("--max-dur", type=float, default=6.0, help="max seconds per cue")
    p.add_argument("--max-gap", type=float, default=0.8,
                   help="silence (s) that forces a new cue")
    p.add_argument("--overwrite", action="store_true",
                   help="re-subtitle files that already have output")

    g = p.add_argument_group("mining")
    g.add_argument("--mine", dest="mine", action="store_true", default=True,
                   help="write a browser page with the clip and hoverable subtitles")
    g.add_argument("--no-mine", dest="mine", action="store_false")
    g.add_argument("--serve", nargs="?", type=int, const=8778, default=None,
                   metavar="PORT",
                   help="serve the mining pages over http with video seeking, "
                        "so Yomitan works without file-URL permissions")

    g = p.add_argument_group("anki cards")
    g.add_argument("--anki", action="store_true",
                   help="also export one card per sentence, with clipped audio")
    g.add_argument("--anki-translate", action="store_true",
                   help="add an English translation to each card")
    g.add_argument("--anki-images", dest="anki_images", action="store_true", default=True,
                   help="grab a screenshot per card")
    g.add_argument("--no-anki-images", dest="anki_images", action="store_false")
    g.add_argument("--anki-image-width", type=int, default=640)
    g.add_argument("--anki-pad", type=float, default=0.25,
                   help="seconds of padding around each audio clip")
    g.add_argument("--anki-media", default=None,
                   help="write media straight into Anki's collection.media")

    args = p.parse_args()
    if not args.files and not args.watch:
        p.error("give some files, or --watch a folder")

    sub = Subtitler(args)

    done = []
    for pattern in args.files:
        path = Path(pattern)
        matches = [path] if path.exists() else sorted(Path().glob(pattern))
        if not matches:
            log("no such file: %s" % pattern)
        for m in matches:
            sub.run(m)
            done.append(m)

    srv = None
    if args.serve:
        folder = Path(args.watch) if args.watch else (
            done[0].parent if done else Path("."))
        srv = serve(folder.resolve(), args.serve)

    if args.watch:
        watch(sub, args.watch, args)
    elif srv is not None:
        log("Ctrl+C to stop serving")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
