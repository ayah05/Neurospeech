from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from neurospeech.evaluation.asr import (
    calculate_asr_metrics,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "abnerh/TORGO-database"
MODEL_NAME = "openai/whisper-small"

METADATA_PATH = (
    ROOT / "data" / "metadata_with_folds.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "whisper_subset_predictions.csv"
)

SUMMARY_PATH = (
    ROOT / "results" / "whisper_subset_summary.txt"
)

RANDOM_STATE = 42

TARGET_PER_CLASS = 50


# ============================================================
# AUDIO HELPER
# ============================================================

def get_waveform_and_sampling_rate(audio):
    """
    Convert the Hugging Face / TorchCodec audio object
    into a mono NumPy waveform and sampling rate.
    """

    if hasattr(audio, "get_all_samples"):

        samples = audio.get_all_samples()

        waveform = samples.data
        sampling_rate = samples.sample_rate

        if hasattr(waveform, "detach"):
            waveform = (
                waveform
                .detach()
                .cpu()
                .numpy()
            )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        # Convert multi-channel audio to mono.
        if waveform.ndim > 1:
            waveform = waveform.mean(
                axis=0
            )

        return waveform, sampling_rate

    raise TypeError(
        "Unsupported audio representation: "
        f"{type(audio)}"
    )


# ============================================================
# BALANCED SPEAKER SAMPLING
# ============================================================

def sample_class_by_speaker(
    class_df,
    target_total,
    random_state,
):
    """
    Sample approximately equally across all speakers
    belonging to one speech-status class.

    Example:
        50 healthy recordings / 7 healthy speakers
        -> roughly 7 recordings per speaker.

    Remaining recordings are distributed one at a time
    across speakers until target_total is reached.
    """

    speakers = sorted(
        class_df["speaker_id"].unique()
    )

    n_speakers = len(speakers)

    base_samples = (
        target_total // n_speakers
    )

    remainder = (
        target_total % n_speakers
    )

    sampled_parts = []

    rng = np.random.default_rng(
        random_state
    )

    # Randomly choose which speakers receive
    # one of the remainder samples.
    remainder_speakers = set(
        rng.choice(
            speakers,
            size=remainder,
            replace=False,
        )
    )

    for i, speaker in enumerate(speakers):

        speaker_df = class_df[
            class_df["speaker_id"]
            == speaker
        ]

        n_samples = base_samples

        if speaker in remainder_speakers:
            n_samples += 1

        if len(speaker_df) < n_samples:
            raise ValueError(
                f"Speaker {speaker} has only "
                f"{len(speaker_df)} recordings, "
                f"but {n_samples} were requested."
            )

        sampled = speaker_df.sample(
            n=n_samples,
            random_state=(
                random_state + i
            ),
        )

        sampled_parts.append(
            sampled
        )

    result = pd.concat(
        sampled_parts,
        ignore_index=True,
    )

    if len(result) != target_total:
        raise ValueError(
            f"Expected {target_total} samples, "
            f"got {len(result)}."
        )

    return result


def create_balanced_subset(metadata):
    """
    Create:
        50 healthy recordings
        50 dysarthria recordings

    while distributing samples approximately equally
    across speakers within each class.
    """

    healthy = metadata[
        metadata["speech_status"]
        == "healthy"
    ].copy()

    dysarthria = metadata[
        metadata["speech_status"]
        == "dysarthria"
    ].copy()

    healthy_sample = (
        sample_class_by_speaker(
            class_df=healthy,
            target_total=TARGET_PER_CLASS,
            random_state=RANDOM_STATE,
        )
    )

    dysarthria_sample = (
        sample_class_by_speaker(
            class_df=dysarthria,
            target_total=TARGET_PER_CLASS,
            random_state=RANDOM_STATE + 100,
        )
    )

    subset = pd.concat(
        [
            healthy_sample,
            dysarthria_sample,
        ],
        ignore_index=True,
    )

    # Shuffle only the processing order.
    subset = subset.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return subset


# ============================================================
# VALIDATE SUBSET
# ============================================================

def validate_subset(subset):

    print("\nSubset validation:")

    if len(subset) != 100:
        raise ValueError(
            f"Expected 100 recordings, "
            f"found {len(subset)}."
        )

    counts = (
        subset["speech_status"]
        .value_counts()
    )

    if counts.get("healthy", 0) != 50:
        raise ValueError(
            "Expected exactly 50 healthy recordings."
        )

    if counts.get("dysarthria", 0) != 50:
        raise ValueError(
            "Expected exactly 50 dysarthria recordings."
        )

    if not subset[
        "dataset_index"
    ].is_unique:
        raise ValueError(
            "Duplicate dataset indices found."
        )

    print("✓ 100 recordings")
    print("✓ 50 healthy")
    print("✓ 50 dysarthria")
    print("✓ No duplicate recordings")

    print("\nSamples per speaker:")

    speaker_counts = (
        subset
        .groupby(
            [
                "speech_status",
                "speaker_id",
            ]
        )
        .size()
        .rename("recordings")
        .reset_index()
    )

    print(
        speaker_counts.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    print(
        "Loading metadata..."
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Total recordings: "
        f"{len(metadata)}"
    )

    print(
        f"Total speakers: "
        f"{metadata['speaker_id'].nunique()}"
    )


    # --------------------------------------------------------
    # CREATE SUBSET
    # --------------------------------------------------------

    print(
        "\nCreating balanced subset..."
    )

    subset = create_balanced_subset(
        metadata
    )

    validate_subset(
        subset
    )


    # --------------------------------------------------------
    # LOAD TORGO
    # --------------------------------------------------------

    print(
        "\nLoading TORGO..."
    )

    dataset = load_dataset(
        DATASET_NAME
    )

    train = dataset["train"]


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = 0

        print(
            "\nUsing GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = -1

        print(
            "\nUsing CPU"
        )


    # --------------------------------------------------------
    # LOAD WHISPER
    # --------------------------------------------------------

    print(
        "\nLoading Whisper..."
    )

    asr = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
    )


    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print(
        "\nRunning Whisper inference..."
    )

    results = []

    for _, row in tqdm(
        subset.iterrows(),
        total=len(subset),
        desc="Whisper",
    ):

        dataset_index = int(
            row["dataset_index"]
        )

        sample = train[
            dataset_index
        ]

        try:

            waveform, sampling_rate = (
                get_waveform_and_sampling_rate(
                    sample["audio"]
                )
            )

            whisper_result = asr(
                {
                    "array": waveform,
                    "sampling_rate":
                        sampling_rate,
                },
                generate_kwargs={
                    "language": "english",
                    "task": "transcribe",
                },
            )

            prediction = (
                whisper_result["text"]
            )

            reference = (
                row["transcription"]
            )

            metrics = (
                calculate_asr_metrics(
                    reference=reference,
                    prediction=prediction,
                )
            )

            result = {
                "dataset_index":
                    dataset_index,

                "speaker_id":
                    row["speaker_id"],

                "speech_status":
                    row["speech_status"],

                "gender":
                    row["gender"],

                "fold":
                    row["fold"],

                "filename":
                    row["filename"],

                "reference":
                    reference,

                "prediction":
                    prediction,

                "reference_normalized":
                    metrics[
                        "reference_normalized"
                    ],

                "prediction_normalized":
                    metrics[
                        "prediction_normalized"
                    ],

                "wer":
                    metrics["wer"],

                "cer":
                    metrics["cer"],

                "error":
                    None,
            }

        except Exception as error:

            print(
                f"\nError for dataset index "
                f"{dataset_index}: "
                f"{error}"
            )

            result = {
                "dataset_index":
                    dataset_index,

                "speaker_id":
                    row["speaker_id"],

                "speech_status":
                    row["speech_status"],

                "gender":
                    row["gender"],

                "fold":
                    row["fold"],

                "filename":
                    row["filename"],

                "reference":
                    row["transcription"],

                "prediction":
                    None,

                "reference_normalized":
                    None,

                "prediction_normalized":
                    None,

                "wer":
                    np.nan,

                "cer":
                    np.nan,

                "error":
                    str(error),
            }

        results.append(
            result
        )


    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    successful = (
        results_df["error"]
        .isna()
        .sum()
    )

    failed = (
        results_df["error"]
        .notna()
        .sum()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WHISPER SUBSET RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSuccessful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )


    # --------------------------------------------------------
    # CLASS SUMMARY
    # --------------------------------------------------------

    successful_df = results_df[
        results_df["error"].isna()
    ].copy()

    class_summary = (
        successful_df
        .groupby("speech_status")
        .agg(
            recordings=(
                "dataset_index",
                "count",
            ),

            mean_wer=(
                "wer",
                "mean",
            ),

            median_wer=(
                "wer",
                "median",
            ),

            mean_cer=(
                "cer",
                "mean",
            ),

            median_cer=(
                "cer",
                "median",
            ),
        )
        .reset_index()
    )

    print(
        "\nResults by speech status:"
    )

    print(
        class_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # --------------------------------------------------------
    # SPEAKER SUMMARY
    # --------------------------------------------------------

    speaker_summary = (
        successful_df
        .groupby(
            [
                "speech_status",
                "speaker_id",
            ]
        )
        .agg(
            recordings=(
                "dataset_index",
                "count",
            ),

            mean_wer=(
                "wer",
                "mean",
            ),

            median_wer=(
                "wer",
                "median",
            ),

            mean_cer=(
                "cer",
                "mean",
            ),

            median_cer=(
                "cer",
                "median",
            ),
        )
        .reset_index()
    )

    print(
        "\nResults by speaker:"
    )

    print(
        speaker_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "NEUROSPEECH WHISPER SUBSET EVALUATION\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(
            f"Model: {MODEL_NAME}\n"
        )

        file.write(
            "Language: English\n"
        )

        file.write(
            "Task: transcription\n"
        )

        file.write(
            "Sample: 50 healthy + "
            "50 dysarthria recordings\n"
        )

        file.write(
            "Sampling: approximately "
            "balanced across speakers\n\n"
        )

        file.write(
            f"Successful: {successful}\n"
        )

        file.write(
            f"Failed: {failed}\n\n"
        )

        file.write(
            "RESULTS BY SPEECH STATUS\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            class_summary.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write(
            "\n\nRESULTS BY SPEAKER\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            speaker_summary.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write("\n")


    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print(
        "\nPredictions saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nSummary saved to:"
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()