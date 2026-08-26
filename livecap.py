#!/usr/bin/env python3
"""
livecap.py - live speech-to-captions for OBS.

Captures audio (WASAPI loopback = "whatever your speakers play", or any mic),
segments it with an energy VAD, transcribes with faster-whisper, and publishes
captions over WebSocket to an OBS Browser Source overlay.

    python livecap.py --list-devices
    python livecap.py --selftest 12
    python livecap.py --lang fi --model small
"""

import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
import wave
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundcard as sc
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
TARGET_SR = 16000
CAPTURE_SR = 48000     # WASAPI shared-mode mix rate on virtually every Windows box
BLOCK_MS = 32

# Whisper invents these when fed silence or noise. Only applied to short results.
HALLUCINATIONS = [
    r"^tekstitys",
    r"^tekstityksen tuotti",
    r"^k[aa]a?nn[oo]s",
    r"^kiitos( kun katsoit| paljon| katsomisesta)?[.!]?$",
    r"^suomennos",
    r"^subtitles? by",
    r"^amara\.org",
    r"^thanks? for watching",
    r"^\W*$",
]
HALLUCINATION_RE = [re.compile(p, re.I) for p in HALLUCINATIONS]


def log(*a):
    print("[" + time.strftime("%H:%M:%S") + "]", *a, flush=True)


# ---------------------------------------------------------------- devices ---

def candidates(loopback):
    """Devices usable in the current mode, in a stable order."""
    mics = sc.all_microphones(include_loopback=True)
    return [m for m in mics if bool(m.isloopback) == bool(loopback)]


def list_devices():
    print("\n=== LOOPBACK sources (default mode: captures what this device plays) ===")
    for i, m in enumerate(candidates(True)):
        print("  %2d  %s  (%dch)" % (i, m.name, m.channels))
    try:
        print("\n  default (used when you pass no --audio-device): %s"
              % sc.default_speaker().name)
    except Exception:
        pass

    print("\n=== INPUT devices (--mic mode: microphones, virtual cables) ===")
    for i, m in enumerate(candidates(False)):
        print("  %2d  %s  (%dch)" % (i, m.name, m.channels))
    try:
        print("\n  default (--mic with no --audio-device): %s" % sc.default_microphone().name)
    except Exception:
        pass
    print("\nSelect with --audio-device followed by the number above, or any\n"
          "case-insensitive part of the name (e.g. --audio-device CABLE).\n")


def resolve_device(spec, loopback):
    """spec: None, an index into the listing, or a substring of the device name."""
    pool = candidates(loopback)
    if not pool:
        raise RuntimeError("No %s devices found." % ("loopback" if loopback else "input"))

    if spec is None:
        try:
            if loopback:
                return sc.get_microphone(sc.default_speaker().id, include_loopback=True)
            return sc.default_microphone()
        except Exception:
            return pool[0]

    try:
        idx = int(spec)
    except ValueError:
        pass
    else:
        if 0 <= idx < len(pool):
            return pool[idx]
        raise RuntimeError("Device index %d out of range (0-%d)." % (idx, len(pool) - 1))

    want = spec.lower()
    for m in pool:
        if want in m.name.lower():
            return m
    raise RuntimeError("No %s device matching %r. Run --list-devices."
                       % ("loopback" if loopback else "input", spec))


# ---------------------------------------------------------------- capture ---

class Capture:
    """Pumps mono float32 blocks off a soundcard device onto a queue."""

    def __init__(self, mic, loopback=True):
        self.mic = mic
        self.name = mic.name
        self.loopback = loopback
        self.sr = CAPTURE_SR
        self.channels = max(1, int(mic.channels))
        self.blocksize = int(self.sr * BLOCK_MS / 1000)
        self.q = queue.Queue(maxsize=256)
        self.dropped = 0
        self.error = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None

    def _pump(self):
        try:
            with self.mic.recorder(samplerate=self.sr, channels=self.channels,
                                   blocksize=self.blocksize) as rec:
                self._ready.set()
                while not self._stop.is_set():
                    data = rec.record(numframes=self.blocksize)
                    if data.ndim > 1 and data.shape[1] > 1:
                        mono = data.mean(axis=1)
                    else:
                        mono = data.reshape(-1)
                    try:
                        self.q.put_nowait(np.ascontiguousarray(mono, dtype=np.float32))
                    except queue.Full:
                        self.dropped += 1
        except Exception as e:
            self.error = e
            log("capture failed:", repr(e))
        finally:
            self._ready.set()

    def __enter__(self):
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self.error is not None:
            raise self.error
        log("capturing: %s  (%dch @ %dHz, %s)"
            % (self.name, self.channels, self.sr, "loopback" if self.loopback else "input"))
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def to_whisper(audio, sr):
    """native-rate mono float32 -> 16 kHz float32, gently gain-staged."""
    if sr != TARGET_SR:
        g = int(np.gcd(sr, TARGET_SR))
        audio = resample_poly(audio, TARGET_SR // g, sr // g)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if 0.0 < peak < 0.35:
        audio = audio * (0.35 / peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# -------------------------------------------------------------- publishing --

class Bus:
    def __init__(self, args):
        self.args = args
        self.loop = None
        self.clients = set()
        self.history = deque(maxlen=12)
        self.finals = deque(maxlen=max(1, args.txt_lines))
        self.txt = HERE / args.txt
        self.logf = HERE / "captions.log"

    def attach(self, loop):
        self.loop = loop

    def publish(self, msg):
        if msg["type"] == "final":
            self.history.append(msg)
            self.finals.append(msg["text"])
            self._write_files(msg)
        if self.loop is not None:
            payload = json.dumps(msg, ensure_ascii=False)
            self.loop.call_soon_threadsafe(self._fanout, payload)

    def _fanout(self, payload):
        for ws in list(self.clients):
            try:
                asyncio.get_running_loop().create_task(ws.send(payload))
            except Exception:
                self.clients.discard(ws)

    def _write_files(self, msg):
        try:
            self.txt.write_text("\n".join(self.finals) + "\n", encoding="utf-8")
            line = time.strftime("%Y-%m-%d %H:%M:%S") + "\t" + msg["text"]
            if msg.get("tr"):
                line += "\t|| " + msg["tr"]
            with self.logf.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as e:
            log("file write failed:", e)


async def ws_server(bus, args, stop):
    import websockets

    async def handler(ws):
        bus.clients.add(ws)
        log("overlay connected (%d client(s))" % len(bus.clients))
        try:
            for m in list(bus.history)[-4:]:
                await ws.send(json.dumps(m, ensure_ascii=False))
            await ws.wait_closed()
        finally:
            bus.clients.discard(ws)
            log("overlay disconnected (%d client(s))" % len(bus.clients))

    async with websockets.serve(handler, "127.0.0.1", args.ws_port, ping_interval=20):
        log("websocket  ws://127.0.0.1:%d" % args.ws_port)
        await stop.wait()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def http_server(args):
    handler = partial(QuietHandler, directory=str(HERE))
    srv = ThreadingHTTPServer(("127.0.0.1", args.http_port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = ("http://127.0.0.1:%d/overlay.html?ws=%d&lines=%d&size=%d&hide=%s"
           % (args.http_port, args.ws_port, args.lines, args.size, args.hide))
    if args.translate:
        url += "&tr=1"
    log("OBS Browser Source URL (copy this):")
    log("  " + url)
    return srv


# ----------------------------------------------------------- transcription --

def looks_hallucinated(text):
    t = text.strip().lower()
    if len(t) > 40:
        return False
    return any(r.search(t) for r in HALLUCINATION_RE)


class Transcriber:
    def __init__(self, args, bus):
        from faster_whisper import WhisperModel

        self.args = args
        self.bus = bus
        log("loading model %r on %s/%s (first run downloads it)..."
            % (args.model, args.compute_device, args.compute))
        t0 = time.time()
        self.model = WhisperModel(
            args.model, device=args.compute_device, compute_type=args.compute,
            cpu_threads=args.threads, num_workers=1)
        log("model ready in %.1fs" % (time.time() - t0))
        self.context = ""
        self.busy = threading.Event()

    def run(self, audio, final):
        a = self.args
        segs, info = self.model.transcribe(
            audio,
            language=None if a.lang == "auto" else a.lang,
            task="transcribe",
            beam_size=a.beam if final else 1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=(self.context or None) if (final and not a.no_context) else None,
            vad_filter=final,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            without_timestamps=True,
        )
        parts, nsp = [], []
        for s in segs:
            parts.append(s.text.strip())
            nsp.append(getattr(s, "no_speech_prob", 0.0))
        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        return text, (max(nsp) if nsp else 1.0), info

    def translate(self, audio):
        segs, _ = self.model.transcribe(
            audio, language=None if self.args.lang == "auto" else self.args.lang,
            task="translate", beam_size=1, temperature=0.0,
            condition_on_previous_text=False, vad_filter=True, without_timestamps=True)
        return re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segs)).strip()

    def worker(self, jobs, stop):
        while not stop.is_set():
            try:
                kind, audio, sr = jobs.get(timeout=0.25)
            except queue.Empty:
                continue
            self.busy.set()
            try:
                t0 = time.time()
                a16 = to_whisper(audio, sr)
                dur = len(a16) / TARGET_SR
                text, nsp, _ = self.run(a16, final=(kind == "final"))
                if not text:
                    if kind == "final":
                        log("no speech recognised in %.1fs segment "
                            "(music/noise, or wrong --lang)" % dur)
                    continue
                if kind == "final":
                    if nsp > 0.75 or looks_hallucinated(text):
                        log("dropped (no_speech=%.2f): %r" % (nsp, text))
                        continue
                    tr = self.translate(a16) if self.args.translate else ""
                    if not self.args.no_context:
                        self.context = (self.context + " " + text)[-220:]
                    took = time.time() - t0
                    log("FINAL %4.1fs audio in %4.1fs (rtf %.2f)  %s"
                        % (dur, took, took / max(dur, 0.01), text))
                    self.bus.publish({"type": "final", "text": text,
                                      "tr": tr, "ts": time.time()})
                else:
                    self.bus.publish({"type": "partial", "text": text, "ts": time.time()})
            except Exception as e:
                log("transcribe error:", repr(e))
            finally:
                self.busy.clear()
                jobs.task_done()


# ---------------------------------------------------------- segmentation ----

def segmenter(cap, tr, jobs, args, stop):
    dur = cap.blocksize / cap.sr
    noise = 1e-4
    preroll = deque(maxlen=max(1, int(0.35 / dur)))
    seg = []
    speech_blocks = 0
    silence = 0.0
    last_partial = 0.0
    last_meter = 0.0

    while not stop.is_set():
        try:
            blk = cap.q.get(timeout=0.3)
        except queue.Empty:
            continue

        rms = float(np.sqrt(np.mean(blk * blk)) + 1e-12)
        if rms < noise:
            noise = 0.90 * noise + 0.10 * rms
        else:
            noise = 0.995 * noise + 0.005 * rms
        gate = max(noise * args.vad_ratio, args.vad_floor)
        voiced = rms > gate

        if args.meter and time.time() - last_meter > 0.25:
            last_meter = time.time()
            db = 20 * np.log10(max(rms, 1e-9))
            bars = int(np.clip((db + 60) / 60 * 40, 0, 40))
            sys.stdout.write("\r%-40s %6.1f dBFS   gate %6.1f   %s"
                             % ("#" * bars, db, 20 * np.log10(gate),
                                "VOICE" if voiced else "     "))
            sys.stdout.flush()

        if not seg:
            if not voiced:
                preroll.append(blk)
                continue
            seg = list(preroll)
            preroll.clear()

        seg.append(blk)
        if voiced:
            speech_blocks += 1
            silence = 0.0
        else:
            silence += dur

        seg_dur = len(seg) * dur
        speech_dur = speech_blocks * dur

        if speech_dur >= args.min_speech and (silence >= args.pause or seg_dur >= args.max_seg):
            if args.verbose:
                log("segment queued: %.1fs (%.1fs of it speech)" % (seg_dur, speech_dur))
            try:
                jobs.put_nowait(("final", np.concatenate(seg), cap.sr))
            except queue.Full:
                log("backlog full - dropping a segment (model too slow for live)")
            seg, speech_blocks, silence = [], 0, 0.0
        elif silence >= args.pause:
            seg, speech_blocks, silence = [], 0, 0.0          # noise blip
        elif (args.partials and speech_dur >= 0.7
              and time.time() - last_partial >= args.partial_every
              and not tr.busy.is_set() and jobs.empty()):
            last_partial = time.time()
            try:
                jobs.put_nowait(("partial", np.concatenate(seg), cap.sr))
            except queue.Full:
                pass


# -------------------------------------------------------------- selftest ----

def selftest(args, seconds):
    dev = resolve_device(args.audio_device, args.loopback)
    with Capture(dev, args.loopback) as cap:
        log("recording %ds -- play / speak Finnish audio NOW..." % seconds)
        blocks, t0, peak = [], time.time(), 0.0
        while time.time() - t0 < seconds:
            try:
                b = cap.q.get(timeout=0.5)
            except queue.Empty:
                continue
            blocks.append(b)
            peak = max(peak, float(np.max(np.abs(b))))
        sr = cap.sr

    if not blocks:
        log("NO AUDIO CAPTURED -- wrong device? Run --list-devices.")
        return 1
    audio = np.concatenate(blocks)
    log("captured %.1fs, peak %.1f dBFS" % (len(audio) / sr, 20 * np.log10(max(peak, 1e-9))))
    if peak < 0.001:
        log("WARNING: that is silence. Pick another device with --audio-device.")

    a16 = to_whisper(audio, sr)
    with wave.open(str(HERE / "selftest.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes((a16 * 32767).astype(np.int16).tobytes())
    log("wrote selftest.wav -- play it to confirm you grabbed the right audio")

    t = Transcriber(args, Bus(args))
    t0 = time.time()
    text, nsp, info = t.run(a16, final=True)
    took = time.time() - t0
    rtf = took / max(len(a16) / TARGET_SR, 0.01)
    log("detected language=%s  no_speech=%.2f" % (getattr(info, "language", "?"), nsp))
    verdict = "FAST ENOUGH for live" if rtf < 0.6 else "TOO SLOW -- use a smaller --model"
    log("transcribed in %.1fs  ->  RTF %.2f  (%s)" % (took, rtf, verdict))
    print("\n  " + (text or "(nothing recognised)") + "\n")
    return 0


# ------------------------------------------------------------------ main ----

def build_parser():
    p = argparse.ArgumentParser(
        description="Live captions for OBS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("audio")
    g.add_argument("--list-devices", action="store_true", help="show devices and exit")
    g.add_argument("--audio-device", default=None, help="index or name substring")
    g.add_argument("--loopback", dest="loopback", action="store_true", default=True,
                   help="capture desktop output (default)")
    g.add_argument("--mic", dest="loopback", action="store_false",
                   help="capture an input device instead of desktop output")
    g.add_argument("--meter", action="store_true", help="print a live level meter")
    g.add_argument("--verbose", action="store_true", help="log VAD segment decisions")

    g = p.add_argument_group("model")
    g.add_argument("--model", default="small",
                   help="tiny|base|small|medium|large-v3|large-v3-turbo, or a local CT2 dir")
    g.add_argument("--lang", default="fi", help="fi, en, sv, ... or 'auto'")
    g.add_argument("--compute-device", default="cpu", choices=["cpu", "cuda"])
    g.add_argument("--compute", default="int8", help="int8|int8_float32|float32|float16")
    g.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 8) - 2))
    g.add_argument("--beam", type=int, default=5)
    g.add_argument("--translate", action="store_true",
                   help="also emit an English translation line")
    g.add_argument("--no-context", action="store_true",
                   help="do not feed previous text back as a prompt")

    g = p.add_argument_group("segmentation")
    g.add_argument("--pause", type=float, default=0.65, help="silence (s) that closes a caption")
    g.add_argument("--min-speech", type=float, default=0.45, help="min speech (s) worth sending")
    g.add_argument("--max-seg", type=float, default=11.0, help="force a cut after this many s")
    g.add_argument("--vad-floor", type=float, default=0.004, help="absolute RMS gate")
    g.add_argument("--vad-ratio", type=float, default=3.0, help="gate = noise floor * this")
    g.add_argument("--partials", dest="partials", action="store_true", default=True)
    g.add_argument("--no-partials", dest="partials", action="store_false",
                   help="only show finished sentences (lower CPU)")
    g.add_argument("--partial-every", type=float, default=0.9)

    g = p.add_argument_group("output")
    g.add_argument("--ws-port", type=int, default=8765)
    g.add_argument("--http-port", type=int, default=8777)
    g.add_argument("--txt", default="captions.txt", help="plain-text file for a GDI+ text source")
    g.add_argument("--txt-lines", type=int, default=2)
    g.add_argument("--lines", type=int, default=2, help="lines shown in the overlay")
    g.add_argument("--size", type=int, default=42, help="overlay font size in px")
    g.add_argument("--hide", type=float, default=8, help="auto-hide overlay after N idle seconds")

    p.add_argument("--selftest", nargs="?", type=int, const=12, default=None, metavar="SECONDS",
                   help="record N seconds, transcribe once, report speed, then exit")
    return p


def main():
    args = build_parser().parse_args()

    if args.list_devices:
        list_devices()
        return 0
    if args.selftest is not None:
        return selftest(args, args.selftest)

    bus = Bus(args)
    stop_ev = threading.Event()
    jobs = queue.Queue(maxsize=3)

    tr = Transcriber(args, bus)
    dev = resolve_device(args.audio_device, args.loopback)

    srv = http_server(args)
    loop = asyncio.new_event_loop()
    bus.attach(loop)
    ws_stop = asyncio.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws_server(bus, args, ws_stop))

    threading.Thread(target=run_loop, daemon=True).start()
    threading.Thread(target=tr.worker, args=(jobs, stop_ev), daemon=True).start()

    try:
        with Capture(dev, args.loopback) as cap:
            log("lang=%s  model=%s  -- Ctrl+C to stop" % (args.lang, args.model))
            segmenter(cap, tr, jobs, args, stop_ev)
    except KeyboardInterrupt:
        print()
        log("stopping")
    finally:
        stop_ev.set()
        loop.call_soon_threadsafe(ws_stop.set)
        srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
