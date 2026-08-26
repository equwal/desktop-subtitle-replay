# desktop-subtitle-replay

Live subtitles for anything playing on your desktop, plus a mining workflow for
the clips you save afterwards. Built for language learning: understand it now,
turn it into Anki cards later.

Runs entirely on your machine. No API keys, no cloud, no internet after the
model downloads.

```
                    ┌─► OBS overlay        (captions on the stream)
desktop audio ──────┼─► reader.html        (selectable text — Yomitan mines it)
   (live)           └─► captions.txt/log   (plain text)

replay clip ────────┬─► clip.srt           (subtitles)
   (offline)        ├─► clip.html          (video + hoverable synced subs)
                    └─► clip.anki.tsv      (one card per sentence + audio)
```

Two halves, because they want opposite things. Live needs speed and accepts
mistakes. Replay needs accuracy and does not care about time.

## Requirements

- Windows 10/11
- Python 3.9–3.12 (3.11 recommended)
- OBS Studio 28+ (only for the live overlay)
- ~2 GB disk for the model cache
- A GPU is *not* required, but it changes what is possible — see below

## Install

```bash
.\setup.ps1
```

## Live captions

```bash
.\run.ps1
```

Prints three URLs:

| page | where it goes |
|---|---|
| `overlay.html` | OBS Browser Source — captions burned into the stream |
| `reader.html` | **your real browser** — selectable text for Yomitan |
| `control.html` | language and model switching while running |

For OBS: **+ → Browser**, paste the overlay URL, **1920 × 1080**, untick
**Shutdown source when not visible**.

For mining: open `reader.html` in the browser where Yomitan is installed. Each
sentence is a plain DOM text node, so Yomitan's popup and its sentence field
work normally. The timestamp is drawn with CSS rather than text, so it never
gets absorbed into the sentence you mine.

### Changing language and model mid-session

Open `control.html`. Language switches on one click or keys `1`–`9` and applies
to the next sentence — no reload. Model switching reloads and pauses captions
for a few seconds. Defaults cover `fi,ru,ja,es,pt,en,auto`:

```bash
.\run.ps1 --langs fi,ru,ja,es,pt,en,auto --lang fi
```

**Whisper has no regional variants.** Argentine Spanish is `es`; there is no
`es-AR`. It handles Rioplatense pronunciation and *voseo*, but normalises
toward standard orthography and will not reliably reproduce regional slang.
Brazilian and European Portuguese are both `pt`.

**`auto` re-detects per segment**, which flips on short or noisy audio and can
mislabel mid-sentence. For deliberate language changes the `1`–`9` keys are far
more reliable.

### Choosing what gets captioned

Default loopback captures everything your speakers play. Your own microphone is
*not* included unless OBS monitors it, and music gets transcribed too.

With [VB-Audio Virtual Cable](https://vb-audio.com/Cable/), send only what you
want captioned:

1. OBS → **Settings → Audio → Advanced → Monitoring Device** = `CABLE Input`
2. Audio Mixer → gear on each source → **Advanced Audio Properties** →
   **Audio Monitoring** = **Monitor and Output**
3. Leave music and alerts on **Monitor Off**

```bash
.\run.ps1 --mic --audio-device CABLE
```

## Subtitling replay clips

```bash
.\.venv\Scripts\python.exe subtitle.py clip.mp4
```

Writes `clip.srt` and `clip.html` next to the clip. The HTML page is the mining
surface: video on the left, every sentence listed as selectable text, click to
seek, `Loop cue` to repeat a line while you work it out.

Watch your replay folder and subtitle clips automatically as OBS saves them:

```bash
.\.venv\Scripts\python.exe subtitle.py --watch "C:\Users\you\Videos" --serve
```

`--serve` matters more than it looks. Opening `clip.html` from `file://`
requires enabling Yomitan's *Allow access to file URLs*, and serving it through
`python -m http.server` **silently breaks video seeking**, because that server
ignores HTTP Range requests. `--serve` runs a range-capable server, so scrubbing
works and Yomitan needs no extra permission.

### Anki cards

```bash
.\.venv\Scripts\python.exe subtitle.py clip.mp4 --anki
```

Produces `clip.anki.tsv` (one row per sentence) plus a media folder of
per-sentence audio clips, with columns: sentence, translation, `[sound:…]`,
`<img>`, source file, timestamp. Copy the media into your Anki
`collection.media` and import the TSV.

This is the batch path. If you mine word-by-word with Yomitan, use `clip.html`
instead and let Yomitan build the cards — it captures the sentence context on
its own, which is usually what you want.

Screenshots need Pillow (`pip install pillow`); without it the image column is
left empty and everything else still works.

## Speed, honestly

Measured on a 16-core CPU, no CUDA, `int8`, on real Finnish speech:

| model | RTF | verdict |
|---|---|---|
| `tiny` | 0.07 | fast, poor Finnish |
| `base` | 0.10 | fast, weak Finnish |
| **`small`** (live default) | **0.57** | the practical ceiling on CPU |
| `large-v3-turbo` (replay default) | 1.77 | too slow live, ideal offline |

RTF is measured over a whole file. **Per segment during a live stream it is
worse** — short utterances pay a fixed encoder cost, so real sessions show 0.55
to 1.4, occasionally decoding slower than real time. Expect captions **3–8 s
behind the speaker**, not 1 s. That is inherent to the approach, not a bug.

Benchmark your own machine:

```bash
.\.venv\Scripts\python.exe bench.py --models small,medium
```

Synthetic audio flatters models because there are fewer tokens to decode. Point
it at a real recording for a number you can trust:

```bash
.\.venv\Scripts\python.exe bench.py --wav selftest.wav
```

### Why it is not truly real-time

Whisper is an offline encoder-decoder over fixed 30-second windows. It cannot
emit a word until it has a chunk to process, so this waits for a pause and
transcribes the finished utterance. That is chunked pseudo-streaming, not
streaming ASR.

Engines that genuinely stream — Kaldi/Vosk online decoding, sherpa-onnx
Zipformer transducers — emit ~200–500 ms after the sound. Vosk covers Russian,
Japanese, Spanish and Portuguese, **but has no Finnish model**. Nothing
genuinely streaming covers this language set; Whisper covers all of it and is
not streaming.

`--stream` implements LocalAgreement streaming (Macháček et al.): re-decode a
growing buffer, commit only the prefix two consecutive decodes agree on. It is
**off by default because it measured worse here on both axes**, on the same
17.6 s Finnish clip:

| mode | speed | output |
|---|---|---|
| chunked `small` | 0.57× | *"Nyt ollaan taas sen verran syrjäisillä seuduilla ja harvakseltaan kuljetuilla seuduilla."* |
| `--stream small` | 3.58× | too slow to run |
| `--stream base` | 0.96× | *"Tolaan taas sen verran Syrjää Näissä ei sillä seudulla…"* |
| `--stream tiny` | 0.34× | *"Kösitäästä. Ja tolaa on… parvaksiautaa"* |

The models fast enough to stream are too weak for Finnish; the model good
enough for Finnish is 3.6× too slow. **With a CUDA GPU this inverts** — run
`--stream --compute-device cuda --compute float16 --model large-v3` and
LocalAgreement becomes the better mode.

To reduce latency without it, shorten the silence needed to close a caption and
free up CPU:

```bash
.\run.ps1 --pause 0.4 --no-partials
```

### Better Finnish at the same speed

Stock `small` is a generalist. A Finnish-fine-tuned `small` is the same size, so
the same speed, but markedly better at Finnish:

```bash
.\get-finnish-model.ps1
```

```bash
.\run.ps1 --model .\build\models\fi-small-ct2
```

Needs ~8 GB free and pulls torch for the one-off conversion; use
`-BuildRoot D:\somewhere` to build elsewhere, then delete `.venv-convert`.

## Syncing captions to the picture

The overlay is composited before any stream delay or replay buffer, so captions
ride along with the video automatically.

To line captions up with lips, delay the picture by the caption latency:

- **Render Delay** filter of `2500`–`4000` ms on video sources
- matching **Sync Offset** on audio sources (Advanced Audio Properties)

If you already run a replay buffer, this costs you nothing.

## Options

`livecap.py`:

| flag | default | effect |
|---|---|---|
| `--lang` / `--langs` | `fi` / `fi,ru,ja,es,pt,en,auto` | active language, and the panel's buttons |
| `--model` | `small` | model name or local CTranslate2 directory |
| `--mic` / `--audio-device` | loopback / auto | capture an input device instead |
| `--pause` | `0.65` | silence that closes a caption; lower is snappier |
| `--min-speech` | `0.45` | ignore bursts shorter than this |
| `--vad-floor` / `--vad-ratio` | `0.004` / `3.0` | speech gate, absolute and relative |
| `--no-partials` | off | finished sentences only; roughly halves CPU |
| `--stream` | off | LocalAgreement streaming (see above) |
| `--translate` | off | add an English line |
| `--compute-device` | `cpu` | `cuda` with an NVIDIA GPU |

`subtitle.py`:

| flag | default | effect |
|---|---|---|
| `--model` | `large-v3-turbo` | accuracy over speed |
| `--watch FOLDER` | — | subtitle clips as they appear |
| `--serve [PORT]` | — | range-capable server so seeking works |
| `--anki` / `--anki-translate` | off | card export, with English backs |
| `--format` | `srt` | `srt`, `vtt`, `txt` |
| `--width` / `--max-chars` | `42` / `84` | line and cue length |

## Diagnostics

```bash
.\run.ps1 --selftest 12
```

Records 12 s, writes `selftest.wav`, transcribes it and reports the real-time
factor. Run this first.

```bash
.\run.ps1 --meter
```

Level meter with the VAD gate, for tuning `--vad-floor`. `--verbose` logs every
segment the VAD sends.

| symptom | cause |
|---|---|
| `no speech recognised in N.Ns segment` | music/noise, or wrong `--lang` |
| nothing in the log at all | VAD never fires — check `--meter` and device |
| `backlog full` | model too slow; use a smaller one |
| video will not seek | server ignoring Range requests — use `--serve` |
| red dot in overlay/reader | lost the WebSocket; it retries automatically |

Whisper hallucinates stock phrases over silence (`Tekstitys: YLE`,
`Kiitos kun katsoit!`, `Thanks for watching`). Short results matching those are
filtered, alongside a no-speech-probability threshold.

## Tests

```bash
.\.venv\Scripts\python.exe tests\test_smoke.py
```

```bash
.\.venv\Scripts\python.exe tests\test_subtitle.py
```

## License

MIT — see [LICENSE](LICENSE). Uses
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) and OpenAI's
Whisper models. Streaming mode follows the LocalAgreement policy from
Macháček, Dabre & Bojar, *Turning Whisper into Real-Time Transcription System*
(2023).
