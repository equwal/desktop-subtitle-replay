# desktop-subtitle-replay

**Live subtitles for anything playing on your screen — then turn what you just heard into Anki cards.**

Watch a stream in Finnish. Read along as it happens. Hit your replay buffer on
the sentence you didn't catch, and get it back as a clip with hoverable
subtitles you can mine with [Yomitan](https://yomitan.wiki/).

Everything runs on your machine. No API keys, no cloud, no upload.

---

## Why this exists

Subtitle tools caption *your* speech for *your* viewers. This does the
opposite: it captions what you're listening to, so you can follow along in a
language you're still learning — and keeps the audio so you can study it later.

**Live and replay want opposite things**, so they get different engines:

|  | live | replay |
|---|---|---|
| needs | speed | accuracy |
| model | `small` (~0.2 RTF) | `large-v3-turbo` |
| output | OBS overlay + reader page | `.srt`, mining page, Anki cards |

The live model transcribed *"harvokseltaan"*. The replay model gets
*"harvakseltaan"* — the actual word. You want both.

## Mining is the point

Yomitan reads **DOM text**, not pixels. So subtitles are rendered as real,
selectable text — each sentence its own text node, timestamps drawn in CSS so
they never contaminate the sentence you mine. Hover a word, get a definition,
make a card, with the sentence context captured automatically.

That works on the live reader *and* on every replay clip.

## Start

```bash
.\setup.ps1
```

```bash
.\start.ps1
```

That's it. Three URLs get printed:

- **reader** — open in the browser where Yomitan lives
- **overlay** — paste into an OBS Browser Source
- **control** — switch language with `1`–`9`, swap models, no restart

Save a replay in OBS and its mining page appears automatically.

## Languages

Any Whisper language. Automatic detection is constrained to the ones you
actually use, so a Finnish clip can't come back as Estonian:

```bash
.\start.ps1 -Lang ja
```

```bash
.\run.ps1 --langs fi,ru,ja,es,pt,en --lang auto
```

## What it won't do

**It runs about 3–8 seconds behind.** Whisper is not a streaming model — it
can't emit a word until it has a chunk to process, so this waits for a pause
and transcribes the finished sentence. Genuinely streaming engines exist, but
none of them cover this language set. If you need lip-sync, delay your video
by the same amount; against a replay buffer that costs nothing.

**It wants CPU.** No GPU required, and `small` holds real time on 16 cores —
but Japanese is tighter than European languages. With a CUDA GPU, everything
here gets better and `--stream` becomes viable.

**It will occasionally invent a sentence.** Whisper hallucinates over silence.
The common stock phrases are filtered, in every language it supports.

---

Full options, tuning, benchmarks and the reasoning behind them:
**[docs/reference.md](docs/reference.md)**

MIT. Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
