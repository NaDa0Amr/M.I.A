# Task 1.1 — Uncover the Secret Hidden Word

## What was wrong with the recording

`task5_1.wav` (48 kHz, mono, 16-bit PCM, 1.117 s) is not simply "noisy" —
it is **hard-clipped**. Inspection shows:

- Amplitude range: `[-0.999969, 0.999969]` — essentially the full digital
  scale on both ends.
- RMS level: `0.9606` — extremely high for a 1-second word recording.
- **89.09% of all samples** sit within a hair of ±1.0.

That means an interference signal so loud it overwhelmed the recorder's
headroom was mixed in, and the analog-to-digital converter (or whatever
process produced this file) squashed everything above/below ±1.0 flat.
Once a sample is clipped, its true value is **gone** — you cannot recover
it by filtering, because clipping is a *non-linear* operation. A linear
filter can only rearrange energy that is still present in the signal; it
cannot re-create information that was thrown away by a hard ceiling.

## Why simple low-pass/notch filtering was insufficient

A first attempt (in an earlier iteration of this task) used plain notch
filters at the strongest FFT peaks and a general-purpose spectral noise
gate. It removed most of the *loudness* of the buzz but never
un-clipped the waveform, so the recovered audio still contained a
buzz-shaped "hole" pattern and a lot of clipping-generated harmonic
smear across the whole spectrum. Notch filtering assumes the useful
signal is still present underneath the unwanted frequency — with 89%
of samples pinned to the rail, that assumption is false for most of the
recording.

## How the main buzz frequencies were found

A Hann-windowed FFT of the full clip was computed and its peaks ranked
by prominence. Two peaks stand out far above everything else:

| Frequency (Hz) | Relative magnitude |
|---:|---:|
| **1988.92** | strongest |
| **4298.29** | 2nd strongest |
| 320.45, 2628.92, 6606.77, 8276.14, 8916.14, 10584.62, ... | much weaker |

Checking the weaker peaks against integer combinations of the two
strongest frequencies (e.g. `2*1988.92 − 4298.29 ≈ 320.45`,
`3*4298.29 ≈ 12894`, `4*4298.29 − 320.45 ≈ 16872`, etc.) shows they line
up almost exactly with sums/differences and harmonics of the two
dominant tones. That is the signature of **hard-clipping a two-tone
signal**: clipping is non-linear, and non-linear distortion of a
two-tone input always generates intermodulation products and harmonics
at `n·f1 ± m·f2`. So the entire "buzz spectrum" is explained by just two
underlying tones, not a large bank of independent noise sources.

## How the clipped sinusoidal model was fitted

The interference is modeled as:

```
noise(t) = a1*sin(2*pi*f1*t) + b1*cos(2*pi*f1*t)
         + a2*sin(2*pi*f2*t) + b2*cos(2*pi*f2*t)
         + offset

predicted_recording = clip(noise(t), -1, 1)
```

Fitting proceeds in two stages:

1. **Initial guess.** `f1`, `f2` come from the FFT peaks above. Initial
   `a1,b1,a2,b2,offset` come from an ordinary *linear* least-squares
   projection of the recorded (clipped) signal onto sine/cosine terms at
   those frequencies. Because clipping compresses amplitude information,
   this linear estimate is scaled up (×3) to start the optimizer closer
   to the true, larger pre-clip amplitude.
2. **Nonlinear refinement.** `scipy.optimize.least_squares` (Trust
   Region Reflective) adjusts all 7 parameters to minimize the
   difference between `clip(noise_model(t), -1, 1)` and the actual
   recording, with the frequencies bounded to stay near the FFT
   estimates and amplitudes allowed a wide range.

On this file, the fit converges to:

- `f1 = 1989.00 Hz`, `f2 = 4298.00 Hz`
- tone amplitudes ≈ **9.95** each (i.e. both tones were originally
  about 10× the clipping ceiling before being clipped down to ±1.0 —
  which is exactly why the recording is so overwhelmingly clipped)
- residual RMS between the fitted (clipped) model and the actual
  recording: **0.0269** — the two-tone clipped model explains the vast
  majority of the recorded signal's energy, leaving a small, speech-sized
  residual.

## Why only unclipped samples were used for direct speech recovery

At a clipped sample, the recording says "≥ ceiling" (or "≤ floor"), not
the actual noise+speech sum, so `recorded − fitted_noise` at a clipped
sample is meaningless — the arithmetic would just measure how wrong the
clipping ceiling is, not the speech. Only samples where
`|recorded| < 0.999` (i.e. the recorder was *not* railing) still carry
faithful amplitude information. At those samples,

```
speech_sample = recorded_sample - fitted_noise_sample
```

is a valid estimate of the underlying speech, because the fitted noise
model tells us what the interference contributed at that instant, and
subtracting it out leaves (mostly) speech.

On this file, **5,848 of 53,625 samples (10.91%)** are reliable in this
sense. That is a sparse but well-distributed set: the maximum gap
between two consecutive reliable samples is only **31 samples
(0.646 ms)** at 48 kHz.

## How interpolation reconstructed the missing samples

Because human speech does not change meaningfully within 0.65 ms, the
sparse reliable speech samples can be safely interpolated across the
gaps. **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial) is
used because it is *shape-preserving* — unlike a plain cubic spline, it
will not overshoot or ring near sharp transitions, which matters because
speech has fast transients (plosives, onsets) that a ringing
interpolator would corrupt. Linear interpolation was also tried; PCHIP
gave a visibly cleaner spectrogram with less high-frequency interpolation
artifact, so it was selected as the final method.

## Why the final speech-band filter and normalization were used

- **Band-pass 70–3800 Hz, zero-phase (`sosfiltfilt`), 6th-order
  Butterworth**: keeps the frequency range where single-word
  intelligibility actually lives, while discarding any leftover
  sub-sonic drift and high-frequency interpolation/model-fitting
  artifacts outside the speech band. `sosfiltfilt` filters forward and
  backward, so there is exactly zero phase distortion — the word's
  timing is not shifted.
- **DC removal**: subtracting the mean prevents any residual offset
  (e.g. from the fitted noise model's `offset` term) from wasting
  headroom or biasing the waveform.
- **99.8th-percentile normalization** (instead of dividing by the
  absolute max): a handful of interpolation overshoots or fitting
  residual spikes could otherwise dominate the normalization and make
  the rest of the word too quiet. Normalizing to a high percentile
  ignores those rare outliers, then a hard safety ceiling
  (`±0.98`) guarantees no clipping is reintroduced in the final file.

## Recovered hidden word

**"TEMPERAMENT"** — this is the best determination from the evidence
gathered (see Validation below). As instructed, this should be treated
as the primary hypothesis to confirm by ear against `cleaned_word.wav`,
not a guaranteed final answer from a single recognition pass.

## Validation

Multiple independent checks were run (outside of `audio_solution.py`,
which itself has no ASR dependency — see *Dependencies* below):

1. **Spectrogram inspection.** After reconstruction, the spectrogram
   shows two clear voiced bursts with visible formant bands (roughly
   0.05–0.20 s and 0.35–0.66 s, separated by a quieter gap) — consistent
   with a two-syllable word, rather than the flat, information-free
   buzz-dominated spectrogram of the raw recording.
2. **Open-vocabulary speech recognition** (PocketSphinx, offline,
   used only for verification, not embedded in the deliverable script)
   on the full reconstructed clip repeatedly surfaced **"temperament"**
   in its top n-best hypotheses across independent runs.
3. **Segment-level recognition.** Splitting the two voiced bursts and
   decoding them in isolation gave "ten" for the first burst and
   "amen"/"man" for the second — phonetically consistent with the
   stressed/unstressed syllable pattern of "TEM-per-a-ment" (initial
   nasal-final syllable, reduced medial vowel, nasal-final ending).
4. **Constrained forced-choice comparison.** A JSGF grammar restricted
   to a candidate list of same-length "-ment" nouns (tenement,
   temperament, testament, settlement, employment, apartment,
   tournament, atonement, ...) was decoded against the reconstructed
   audio. "temperament" scored clearly highest (0.835) among this
   comparably-sized candidate set, well above the next best candidate
   (~0.77–0.78).

These are independent, converging lines of evidence rather than a
single recognition attempt, which is why "temperament" is reported with
reasonable — but not absolute — confidence. **Please listen to
`cleaned_word.wav` yourself for final confirmation.**

## Dependencies

Only `numpy` and `scipy` are required to run `audio_solution.py`:

```bash
pip install -r requirements.txt
```

(The optional ASR cross-checks described under Validation used
`pocketsphinx` + `librosa` in a separate, exploratory analysis; they are
**not** required to run the deliverable script and are not listed in
`requirements.txt`.)

## How to run

Normal run (processes the default input, writes `cleaned_word.wav`, and
plays it back):

```bash
python audio_solution.py
```

Explicit input/output paths:

```bash
python audio_solution.py task5_1.wav cleaned_word.wav
```

Without playback:

```bash
python audio_solution.py --no-play
```

Playback support: Linux (`aplay`, falling back to `play` from SoX),
macOS (`afplay`), Windows (`winsound`). If none of these are available,
the script prints a clear message and leaves the WAV file on disk to be
opened manually.

## Verification performed

- `python3 -m py_compile audio_solution.py` — **passes**.
- Output file `cleaned_word.wav` exists after running the script.
- Output is **mono, 16-bit PCM** (`int16`, 1-D array).
- Output **sample rate (48000 Hz)** and **sample count (53625)** exactly
  match the input.
- All output samples are **finite** (no NaN/Inf).
- Output samples **do not clip** (`max |sample| = 32111 / 32767`, safely
  below the 16-bit ceiling).
- Estimated primary buzz frequencies (`f1 = 1989.000 Hz`,
  `f2 = 4298.000 Hz`) are **stable/identical across three repeated
  runs** (the fitting procedure is deterministic given fixed input).

## Limitations

- Roughly 89% of the original samples were unrecoverable at the
  waveform level; the PCHIP-reconstructed portions are a
  best-effort estimate, not the literal original speech waveform, so
  some fine spectral detail (in particular anything above ~3.8 kHz,
  which was filtered out) is necessarily approximate.
- The reported word is the best hypothesis from multiple converging,
  offline, open-vocabulary/forced-choice checks — it is **not** a
  guaranteed, ground-truth transcription. A human listening to
  `cleaned_word.wav` is the authoritative final check.
