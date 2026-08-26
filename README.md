# livecap — live captions for OBS

Real-time speech captions for OBS Studio on Windows. Captures audio, detects
speech, transcribes it with Whisper, and renders it into a transparent Browser
Source overlay.

Runs entirely on your machine. No API keys, no cloud, no internet after the
model is downloaded. Built for Finnish, works with any language Whisper
supports.

```
audio device ──► VAD segmenter ──► faster-whisper ──► WebSocket ──► overlay.html
                                                  └─► captions.txt (GDI+ fallback)
```

## Requirements

- Windows 10/11
- Python 3.9–3.12 (3.11 recommended)
- OBS Studio 28+
- ~2 GB disk for the model cache
- A GPU is *not* required — this is tuned to run on CPU

## Install

```bash
.\setup.ps1
```

Creates a `.venv` and installs dependencies. One-time.

## Run

```bash
.\run.ps1
```

Prints a Browser Source URL. In OBS: **+ → Browser**, paste it, set
**1920 × 1080**, and untick **Shutdown source when not visible**.

```
http://127.0.0.1:8777/overlay.html?ws=8765&lines=2&size=42&hide=8
```

The overlay is transparent and places captions in the lower third. Keep the
console window open while streaming; Ctrl+C stops it.

Check your devices first if needed:

```bash
.\run.ps1 --list-devices
```

## Changing language and model while streaming

`run.ps1` also prints a control panel URL. Open it in any browser (a second
monitor, a phone on the same machine, or an OBS dock via
**Docks → Custom Browser Docks**):

```
http://127.0.0.1:8777/control.html?ws=8765
```

- **Language** — one click, or press `1`–`9`. Applies to the next sentence.
  No restart and no model reload, so it is effectively instant.
- **Model** — a dropdown. Swapping reloads the model, which pauses captions for
  a few seconds; the panel shows a *loading* state while it happens.
- **Toggles** — English translation, live partial text, and clear captions.
- A live feed of what is being recognised, so you can sanity-check without
  looking at the stream.

Choose which languages get buttons:

```bash
.\run.ps1 --langs fi,ru,ja,es,pt,en,auto --lang fi
```

That default set covers Finnish, Russian, Japanese, Spanish, Portuguese,
English and auto-detect. Any [Whisper language code](https://github.com/openai/whisper#available-models-and-languages)
works.

Two things worth knowing:

- **Whisper has no regional variants.** Argentine Spanish is `es` — there is no
  `es-AR`. The model handles Rioplatense pronunciation and *voseo* as part of
  `es`, but it normalises toward standard orthography and will not reliably
  reproduce regional slang. The same applies to Brazilian vs. European
  Portuguese: both are `pt`.
- **`auto` re-detects per segment**, which sounds ideal for multilingual streams
  but flips on short or noisy utterances and can mislabel mid-sentence. For a
  stream that switches language deliberately, the `1`–`9` keys are far more
  reliable than `auto`.

Wider models are better at non-English audio. If you mostly caption Russian or
Japanese, `medium` is worth the latency if your CPU can take it — check with
`bench.py` first.

## Choosing what gets captioned

**Default (`--loopback`)** captures everything your speakers play — guests,
video, game audio, music. Zero setup. The catch: your own microphone is *not*
included unless OBS monitors it, and music gets fed to Whisper too.

**Recommended: route a dedicated mix through a virtual cable.** With
[VB-Audio Virtual Cable](https://vb-audio.com/Cable/) installed:

1. OBS → **Settings → Audio → Advanced → Monitoring Device** =
   `CABLE Input (VB-Audio Virtual Cable)`
2. Audio Mixer → gear icon on each source you want captioned →
   **Advanced Audio Properties** → **Audio Monitoring** = **Monitor and Output**
   (this keeps the source audible to viewers; *Monitor Only* would mute it)
3. Leave music, game and alert sources on **Monitor Off**
4. Run:

```bash
.\run.ps1 --mic --audio-device CABLE
```

Accuracy improves noticeably once Whisper stops trying to transcribe your
background music.

## Model selection

Measured on a 16-core CPU with no CUDA GPU, `int8` quantisation:

| model | real speech RTF | latency per segment | Finnish quality |
|---|---|---|---|
| `tiny` | 0.07 | ~0.4 s | poor |
| `base` | 0.10 | ~0.6 s | weak |
| **`small`** (default) | **0.57** | **~2.5 s** | good |
| `large-v3-turbo` | 1.77 | ~10.6 s | unusable — falls behind |

RTF (real-time factor) below ~0.6 keeps up with continuous speech. Without a
CUDA GPU, `small` is the largest model that stays real-time, which is why it is
the default. With an NVIDIA GPU, add `--compute-device cuda --compute float16`
and `large-v3` becomes viable.

Benchmark your own machine:

```bash
.\.venv\Scripts\python.exe .\bench.py --models small,medium
```

Synthetic audio makes models look faster than they are, because there are fewer
tokens to decode. For a realistic number, point it at a recording:

```bash
.\.venv\Scripts\python.exe .\bench.py --wav selftest.wav
```

### Better Finnish at the same speed

Stock `small` is a generalist. A Finnish-fine-tuned `small` is the same size —
so the same speed — but markedly better at Finnish.
`get-finnish-model.ps1` fetches one and converts it to CTranslate2:

```bash
.\get-finnish-model.ps1
```

```bash
.\run.ps1 --model .\build\models\fi-small-ct2
```

The conversion needs ~8 GB free and pulls in torch, which is used only for that
one-off step. Use `-BuildRoot D:\somewhere` to build on another drive, and
delete `.venv-convert` under the build root afterwards to reclaim the space.

## Caption latency and stream delay

The overlay is composited into the program feed *before* any stream delay or
replay buffer, so captions ride along with the video automatically — a buffer
needs no special handling.

A caption appears roughly **2.5 s after the sentence ends**: inference time plus
the 0.65 s of silence used to detect the end of the sentence. To lock captions
to lips, delay the picture to match:

- **Render Delay** filter of `2500` ms on your video sources
- **Sync Offset** of `2500` ms on the audio sources
  (Advanced Audio Properties)

If you already run a replay buffer or stream delay, the extra 2.5 s costs you
nothing.

## Options

| flag | default | effect |
|---|---|---|
| `--lang` | `fi` | `fi`, `en`, `sv`, … or `auto` |
| `--model` | `small` | model name or a local CTranslate2 directory |
| `--mic` | off | capture an input device instead of desktop output |
| `--audio-device` | auto | index from `--list-devices`, or part of the name |
| `--pause` | `0.65` | silence (s) that ends a caption; lower = snappier, more fragments |
| `--min-speech` | `0.45` | ignore bursts shorter than this |
| `--max-seg` | `11.0` | force a cut during non-stop speech |
| `--vad-floor` | `0.004` | absolute level gate; raise if noise triggers captions |
| `--vad-ratio` | `3.0` | gate relative to the running noise floor |
| `--no-partials` | off | only show finished sentences; roughly halves CPU |
| `--translate` | off | add an English line beneath the original |
| `--lines` / `--size` | `2` / `42` | overlay line count and font size |
| `--hide` | `8` | fade the overlay out after N idle seconds |
| `--compute-device` | `cpu` | `cuda` if you have an NVIDIA GPU |

## Diagnostics

```bash
.\run.ps1 --selftest 12
```

Records 12 s, writes `selftest.wav`, transcribes it once and reports the
real-time factor. Run this first — speak or play audio during those 12 seconds
and check what comes back.

```bash
.\run.ps1 --meter
```

Live level meter showing the VAD gate, for tuning `--vad-floor`. Add
`--verbose` to log every segment the VAD decides to send.

| symptom | cause |
|---|---|
| `no speech recognised in N.Ns segment` | music/noise reaching Whisper, or wrong `--lang` |
| nothing at all in the log | VAD never fires — check `--meter`, wrong device |
| `backlog full` | model too slow for real time; use a smaller one |
| red dot in the overlay | overlay lost the WebSocket; it retries automatically |

Whisper likes to hallucinate stock phrases over silence (`Tekstitys: YLE`,
`Kiitos kun katsoit!`, `Thanks for watching`). Short results matching those are
filtered out, alongside a no-speech-probability threshold.

## Overlay styling

Append query parameters to the Browser Source URL:

| param | example | effect |
|---|---|---|
| `size` | `size=52` | font size in px |
| `lines` | `lines=3` | how many lines stay on screen |
| `align` | `align=top` | `top`, `center`, default bottom |
| `fg` | `fg=ffe066` | text colour (hex, no `#`) |
| `accent` | `accent=7dd3fc` | translation line colour |
| `box` | `box=0` | remove the dark background pill |
| `hide` | `hide=0` | never auto-hide |
| `tr` | `tr=1` | show translation lines (with `--translate`) |

## Text output

`captions.txt` holds the last few lines for an OBS **Text (GDI+)** source with
*Read from file*, if you would rather avoid browser sources. `captions.log`
keeps the full timestamped transcript of the session, which doubles as stream
notes.

## How it works

`livecap.py` pulls 32 ms blocks from a WASAPI device via `soundcard`, tracks a
running noise floor, and marks blocks as speech when they exceed
`max(noise × ratio, floor)`. Speech accumulates into a segment; a segment closes
on `--pause` of trailing silence or at `--max-seg`. Closed segments go to
faster-whisper on a worker thread and are published as `final`. While a segment
is still open, idle worker time is spent transcribing the partial buffer and
publishing lower-confidence `partial` text, so captions appear before the
speaker finishes. Results are broadcast over WebSocket to `overlay.html` and
mirrored to disk.

## License

MIT — see [LICENSE](LICENSE).

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) and
OpenAI's Whisper models.
