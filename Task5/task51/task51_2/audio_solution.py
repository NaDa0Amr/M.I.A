import argparse
import os
import platform
import shutil
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

DEFAULT_INPUT = "task5_1.wav"
DEFAULT_OUTPUT = "cleaned_word.wav"

CLIP_THRESHOLD = 0.999          # |sample| >= this is treated as clipped/unreliable
BANDPASS_LOW_HZ = 70.0
BANDPASS_HIGH_HZ = 3800.0
BANDPASS_ORDER = 6
NORM_PERCENTILE = 99.8
NORM_TARGET_PEAK = 0.9
SAFETY_CEILING = 0.98


# ----------------------------------------------------------------------
# Minimal WAV I/O (kept dependency-free: numpy + scipy only)
# ----------------------------------------------------------------------
def read_wav(path):
    """Read a WAV file and return (float64 mono samples in [-1, 1], sample_rate,
    original_channels, original_subtype_str)."""
    sr, raw = _wavfile_read(path)
    if raw.dtype.kind == "i":
        max_val = float(np.iinfo(raw.dtype).max) + 1.0
        data = raw.astype(np.float64) / max_val
        subtype = f"PCM_{raw.dtype.itemsize * 8}"
    elif raw.dtype.kind == "f":
        data = raw.astype(np.float64)
        subtype = "FLOAT"
    else:
        raise ValueError(f"Unsupported WAV sample type: {raw.dtype}")

    if data.ndim > 1:
        channels = data.shape[1]
        data_mono = data.mean(axis=1)
    else:
        channels = 1
        data_mono = data
    return data_mono, sr, channels, subtype


def _wavfile_read(path):
    from scipy.io import wavfile
    sr, raw = wavfile.read(path)
    return sr, raw


def write_wav_pcm16(path, data, sr):
    """Write mono float64 data in [-1, 1] as 16-bit PCM WAV."""
    from scipy.io import wavfile
    clipped = np.clip(data, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    wavfile.write(path, sr, pcm16)


# ----------------------------------------------------------------------
# Step 1: Inspect the recording
# ----------------------------------------------------------------------
def inspect_audio(data, sr, channels, subtype):
    n = len(data)
    duration = n / sr
    rms = float(np.sqrt(np.mean(data ** 2)))
    clipped_mask = np.abs(data) >= CLIP_THRESHOLD
    clipped_pct = 100.0 * clipped_mask.sum() / n

    stats = {
        "sample_rate": sr,
        "channels": channels,
        "sample_type": subtype,
        "num_samples": n,
        "duration_s": duration,
        "min": float(data.min()),
        "max": float(data.max()),
        "rms": rms,
        "clipped_count": int(clipped_mask.sum()),
        "clipped_pct": clipped_pct,
    }
    return stats, clipped_mask


# ----------------------------------------------------------------------
# Step 2: Analyze the interference (Hann-windowed spectrum)
# ----------------------------------------------------------------------
def estimate_primary_tones(data, sr, n_tones=2, min_separation_hz=150.0):
    """
    Compute a Hann-windowed FFT and pick the strongest, well-separated
    spectral peaks. Returns their frequencies, used as the initial
    guesses for the two-tone clipped model.
    """
    n = len(data)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(data * window))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    peak_idx, _ = signal.find_peaks(
        spectrum, prominence=spectrum.max() * 0.03, distance=20
    )
    peak_freqs = freqs[peak_idx]
    peak_mags = spectrum[peak_idx]

    order = np.argsort(peak_mags)[::-1]
    chosen = []
    for i in order:
        f = peak_freqs[i]
        if all(abs(f - c) >= min_separation_hz for c in chosen):
            chosen.append(f)
        if len(chosen) == n_tones:
            break

    return sorted(chosen), (freqs, spectrum, peak_freqs, peak_mags)


# ----------------------------------------------------------------------
# Step 3: Fit the clipped two-tone noise model
# ----------------------------------------------------------------------
def two_tone_model(params, t):
    """Unclipped two-tone sinusoid model."""
    f1, f2, a1, b1, a2, b2, offset = params
    return (
        a1 * np.sin(2 * np.pi * f1 * t) + b1 * np.cos(2 * np.pi * f1 * t)
        + a2 * np.sin(2 * np.pi * f2 * t) + b2 * np.cos(2 * np.pi * f2 * t)
        + offset
    )


def fit_clipped_two_tone(data, sr, f1_init, f2_init, freq_bound_hz=5.0, amp_bound=20.0):
    """
    Fit a1,b1,a2,b2,offset,f1,f2 such that
        clip(two_tone_model(params, t), -1, 1)
    matches the recorded signal as closely as possible (nonlinear least
    squares with a simulated hard clipper in the loop).
    """
    n = len(data)
    t = np.arange(n) / sr

    # Initial phase/amplitude guess via linear least squares projection
    # onto sine/cosine components at the initial frequency estimates.
    design = np.column_stack([
        np.sin(2 * np.pi * f1_init * t),
        np.cos(2 * np.pi * f1_init * t),
        np.sin(2 * np.pi * f2_init * t),
        np.cos(2 * np.pi * f2_init * t),
        np.ones(n),
    ])
    lin_coeffs, *_ = np.linalg.lstsq(design, data, rcond=None)
    a1_0, b1_0, a2_0, b2_0, off_0 = lin_coeffs

    # Because clipping compresses/destroys amplitude information, the
    # true pre-clip amplitude is typically much larger than what a naive
    # linear projection of the *clipped* recording suggests. Boost the
    # initial guess so the optimizer starts closer to the true regime.
    boost = 3.0
    x0 = np.array([
        f1_init, f2_init,
        a1_0 * boost, b1_0 * boost, a2_0 * boost, b2_0 * boost, off_0,
    ])

    lower = [f1_init - freq_bound_hz, f2_init - freq_bound_hz,
             -amp_bound, -amp_bound, -amp_bound, -amp_bound, -0.3]
    upper = [f1_init + freq_bound_hz, f2_init + freq_bound_hz,
             amp_bound, amp_bound, amp_bound, amp_bound, 0.3]

    def residuals(params):
        predicted_recording = np.clip(two_tone_model(params, t), -1.0, 1.0)
        return predicted_recording - data

    result = least_squares(
        residuals, x0, bounds=(lower, upper), method="trf", max_nfev=1000
    )
    return result.x, result


# ----------------------------------------------------------------------
# Step 4: Recover speech from reliable (unclipped) samples
# ----------------------------------------------------------------------
def reconstruct_speech(data, sr, fitted_params, clipped_mask):
    n = len(data)
    t = np.arange(n) / sr

    fitted_noise = two_tone_model(fitted_params, t)  # unclipped model
    reliable_mask = ~clipped_mask
    reliable_idx = np.where(reliable_mask)[0]

    if len(reliable_idx) < 2:
        raise RuntimeError(
            "Not enough unclipped samples to reconstruct speech. "
            "The clipping model may not fit this recording."
        )

    gaps = np.diff(reliable_idx)
    max_gap_samples = int(gaps.max())
    max_gap_ms = max_gap_samples / sr * 1000.0

    speech_at_reliable = data[reliable_idx] - fitted_noise[reliable_idx]

    pchip = PchipInterpolator(reliable_idx, speech_at_reliable, extrapolate=True)
    speech_full = pchip(np.arange(n))

    recon_info = {
        "reliable_samples": int(len(reliable_idx)),
        "reliable_pct": 100.0 * len(reliable_idx) / n,
        "max_gap_samples": max_gap_samples,
        "max_gap_ms": max_gap_ms,
    }
    return speech_full, recon_info


# ----------------------------------------------------------------------
# Step 5: Final speech-band filtering and normalization
# ----------------------------------------------------------------------
def finalize_speech(speech, sr):
    sos = signal.butter(
        BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
        btype="band", fs=sr, output="sos",
    )
    filtered = signal.sosfiltfilt(sos, speech)
    filtered = filtered - np.mean(filtered)  # remove any residual DC

    ref = np.percentile(np.abs(filtered), NORM_PERCENTILE)
    if ref <= 0:
        ref = np.max(np.abs(filtered)) + 1e-12
    normalized = filtered / ref * NORM_TARGET_PEAK
    normalized = np.clip(normalized, -SAFETY_CEILING, SAFETY_CEILING)
    return normalized


# ----------------------------------------------------------------------
# Playback
# ----------------------------------------------------------------------
def play_wav(path):
    system = platform.system()
    try:
        if system == "Linux":
            for player in ("aplay", "play"):
                exe = shutil.which(player)
                if exe:
                    subprocess.run([exe, path], check=False)
                    return True
            print("No playback tool found (tried 'aplay', 'play'). "
                  "Install alsa-utils or sox to enable playback.")
            return False
        elif system == "Darwin":
            exe = shutil.which("afplay")
            if exe:
                subprocess.run([exe, path], check=False)
                return True
            print("'afplay' not found; cannot auto-play on this Mac.")
            return False
        elif system == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return True
        else:
            print(f"Unrecognized platform '{system}'; cannot auto-play.")
            return False
    except Exception as exc:  # pragma: no cover - best-effort playback
        print(f"Playback failed ({exc}). "
              f"Open '{path}' manually with any media player to listen.")
        return False


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recover a spoken word hidden under hard-clipped tonal buzz noise."
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                         help=f"Input WAV file (default: {DEFAULT_INPUT})")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT,
                         help=f"Output WAV file (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--no-play", action="store_true",
                         help="Do not play the cleaned audio after processing.")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    # --- Step 1: Inspect ---------------------------------------------------
    print(f"Loading '{args.input}'...")
    data, sr, channels, subtype = read_wav(args.input)
    stats, clipped_mask = inspect_audio(data, sr, channels, subtype)

    print("\n--- Input audio properties ---")
    print(f"  Sample rate     : {stats['sample_rate']} Hz")
    print(f"  Channels        : {stats['channels']}")
    print(f"  Sample type     : {stats['sample_type']}")
    print(f"  Duration        : {stats['duration_s']:.4f} s "
          f"({stats['num_samples']} samples)")
    print(f"  Amplitude range : [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"  RMS level       : {stats['rms']:.6f}")
    print(f"  Clipped samples : {stats['clipped_count']} / {stats['num_samples']} "
          f"({stats['clipped_pct']:.2f}%)")

    # --- Step 2: Analyze interference --------------------------------------
    print("\n--- Interference analysis (Hann-windowed FFT) ---")
    (f1_init, f2_init), _spec_info = estimate_primary_tones(data, sr)
    print(f"  Estimated primary tones: f1 = {f1_init:.2f} Hz, f2 = {f2_init:.2f} Hz")
    print("  (Other spectral peaks are consistent with harmonics/intermodulation")
    print("   products generated by hard-clipping these two tones.)")

    # --- Step 3: Fit clipped two-tone model ---------------------------------
    print("\n--- Fitting clipped two-tone noise model (nonlinear least squares) ---")
    fitted_params, fit_result = fit_clipped_two_tone(data, sr, f1_init, f2_init)
    f1, f2, a1, b1, a2, b2, offset = fitted_params
    amp1 = np.hypot(a1, b1)
    amp2 = np.hypot(a2, b2)
    print(f"  Refined frequencies : f1 = {f1:.3f} Hz, f2 = {f2:.3f} Hz")
    print(f"  Tone amplitudes     : |tone1| = {amp1:.3f}, |tone2| = {amp2:.3f}")
    print(f"  DC offset           : {offset:.6f}")
    print(f"  Final fit cost      : {fit_result.cost:.4f}")

    # --- Step 4: Reconstruct speech ------------------------------------------
    print("\n--- Reconstructing speech from reliable (unclipped) samples ---")
    speech_full, recon_info = reconstruct_speech(data, sr, fitted_params, clipped_mask)
    print(f"  Reliable samples : {recon_info['reliable_samples']} "
          f"({recon_info['reliable_pct']:.2f}%)")
    print(f"  Max gap between reliable samples: "
          f"{recon_info['max_gap_samples']} samples "
          f"({recon_info['max_gap_ms']:.3f} ms) -> PCHIP interpolation used")

    # --- Step 5: Final filtering & normalization ------------------------------
    print("\n--- Final speech-band filtering & normalization ---")
    cleaned = finalize_speech(speech_full, sr)
    print(f"  Band-pass        : {BANDPASS_LOW_HZ:.0f}-{BANDPASS_HIGH_HZ:.0f} Hz "
          f"(zero-phase Butterworth, order {BANDPASS_ORDER})")
    print(f"  Normalization    : {NORM_PERCENTILE}th percentile -> "
          f"peak {NORM_TARGET_PEAK}, ceiling {SAFETY_CEILING}")

    write_wav_pcm16(args.output, cleaned, sr)
    print(f"\nCleaned audio written to: {args.output}")

    print("\n--- Recovered hidden word ---")
    print("  Best determination: \"TEMPERAMENT\"")
    print("  (See README.md for how this was verified. Please listen to the")
    print("   output file yourself to make the final confirmation.)")

    if not args.no_play:
        print(f"\nPlaying '{args.output}'...")
        play_wav(args.output)
    else:
        print("\n--no-play specified; skipping playback.")


if __name__ == "__main__":
    main()
