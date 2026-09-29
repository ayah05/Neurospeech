import numpy as np
import librosa


def extract_acoustic_features(
    waveform,
    sampling_rate,
    frame_length=2048,
    hop_length=512,
    top_db=30,
):
    """
    Extract interpretable acoustic features from one audio recording.

    Features:
        - duration
        - voiced duration
        - speech ratio
        - pause ratio
        - number of pauses
        - mean pause duration
        - mean F0
        - F0 standard deviation
        - mean RMS energy
        - RMS energy standard deviation

    Parameters
    ----------
    waveform:
        Audio waveform as a NumPy array.

    sampling_rate:
        Sampling rate in Hz.

    frame_length:
        Frame size used for energy-based speech detection.

    hop_length:
        Number of samples between analysis frames.

    top_db:
        Signals more than `top_db` dB below the reference
        are considered silence by librosa.effects.split().

    Returns
    -------
    dict
        Dictionary containing acoustic features.
    """

    waveform = np.asarray(waveform, dtype=np.float32)

    # Handle stereo/multi-channel audio if encountered.
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    # Remove NaN/Inf values if present.
    waveform = np.nan_to_num(waveform)

    # ---------------------------------------------------------
    # DURATION
    # ---------------------------------------------------------

    duration = len(waveform) / sampling_rate

    # ---------------------------------------------------------
    # SPEECH / SILENCE SEGMENTATION
    # ---------------------------------------------------------

    non_silent_intervals = librosa.effects.split(
        waveform,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    voiced_samples = sum(
        end - start
        for start, end in non_silent_intervals
    )

    voiced_duration = voiced_samples / sampling_rate

    if duration > 0:
        speech_ratio = voiced_duration / duration
        pause_ratio = 1.0 - speech_ratio
    else:
        speech_ratio = 0.0
        pause_ratio = 0.0

    # ---------------------------------------------------------
    # PAUSES
    # ---------------------------------------------------------

    pause_durations = []

    if len(non_silent_intervals) > 1:

        for i in range(len(non_silent_intervals) - 1):

            current_end = non_silent_intervals[i][1]
            next_start = non_silent_intervals[i + 1][0]

            pause_samples = next_start - current_end
            pause_duration = pause_samples / sampling_rate

            if pause_duration > 0:
                pause_durations.append(pause_duration)

    number_of_pauses = len(pause_durations)

    mean_pause_duration = (
        float(np.mean(pause_durations))
        if pause_durations
        else 0.0
    )

    # ---------------------------------------------------------
    # FUNDAMENTAL FREQUENCY (F0)
    # ---------------------------------------------------------

    try:

        f0 = librosa.yin(
            waveform,
            fmin=50,
            fmax=500,
            sr=sampling_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        valid_f0 = f0[
            np.isfinite(f0)
            & (f0 >= 50)
            & (f0 <= 500)
        ]

        if len(valid_f0) > 0:
            f0_mean = float(np.mean(valid_f0))
            f0_std = float(np.std(valid_f0))
        else:
            f0_mean = np.nan
            f0_std = np.nan

    except Exception:
        f0_mean = np.nan
        f0_std = np.nan

    # ---------------------------------------------------------
    # RMS ENERGY
    # ---------------------------------------------------------

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if len(rms) > 0:
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
    else:
        rms_mean = np.nan
        rms_std = np.nan

    # ---------------------------------------------------------
    # RETURN FEATURES
    # ---------------------------------------------------------

    return {
        "audio_duration": float(duration),
        "voiced_duration": float(voiced_duration),
        "speech_ratio": float(speech_ratio),
        "pause_ratio": float(pause_ratio),
        "number_of_pauses": int(number_of_pauses),
        "mean_pause_duration": mean_pause_duration,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
    }