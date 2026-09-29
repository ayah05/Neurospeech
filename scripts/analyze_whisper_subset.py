from pathlib import Path

import numpy as np
import pandas as pd
from jiwer import wer, cer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    ROOT / "data" / "whisper_subset_predictions.csv"
)

RESULTS_DIR = ROOT / "results"

REPORT_PATH = (
    RESULTS_DIR / "whisper_subset_analysis.txt"
)

SPEAKER_RESULTS_PATH = (
    RESULTS_DIR / "whisper_subset_speaker_results.csv"
)

WORST_CASES_PATH = (
    RESULTS_DIR / "whisper_subset_worst_cases.csv"
)


# ============================================================
# CONFIG
# ============================================================

N_WORST_CASES = 10


# ============================================================
# CORPUS-LEVEL METRICS
# ============================================================

def calculate_corpus_metrics(data):
    """
    Calculate corpus-level WER and CER.

    Instead of averaging WER/CER across recordings,
    all normalized references and predictions are
    combined first.

    This means longer utterances contribute
    proportionally more linguistic content.
    """

    valid = data[
        data["reference_normalized"].notna()
        & data["prediction_normalized"].notna()
    ].copy()

    references = (
        valid["reference_normalized"]
        .astype(str)
        .tolist()
    )

    predictions = (
        valid["prediction_normalized"]
        .astype(str)
        .tolist()
    )

    corpus_wer = wer(
        references,
        predictions,
    )

    # For CER we join utterances using newlines.
    # This keeps utterance boundaries visible while
    # evaluating the full character stream.
    reference_text = "\n".join(
        references
    )

    prediction_text = "\n".join(
        predictions
    )

    corpus_cer = cer(
        reference_text,
        prediction_text,
    )

    return {
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
    }


# ============================================================
# UTTERANCE-LEVEL SUMMARY
# ============================================================

def calculate_utterance_summary(data):
    """
    Mean and median of per-recording WER/CER.
    """

    return (
        data
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
            std_wer=(
                "wer",
                "std",
            ),
            mean_cer=(
                "cer",
                "mean",
            ),
            median_cer=(
                "cer",
                "median",
            ),
            std_cer=(
                "cer",
                "std",
            ),
        )
        .reset_index()
    )


# ============================================================
# SPEAKER-LEVEL ANALYSIS
# ============================================================

def calculate_speaker_metrics(data):
    """
    Calculate utterance-level and corpus-level metrics
    separately for each speaker.
    """

    rows = []

    for (
        speech_status,
        speaker_id,
    ), speaker_df in data.groupby(
        [
            "speech_status",
            "speaker_id",
        ]
    ):

        corpus_metrics = (
            calculate_corpus_metrics(
                speaker_df
            )
        )

        row = {
            "speech_status":
                speech_status,

            "speaker_id":
                speaker_id,

            "recordings":
                len(speaker_df),

            "mean_wer":
                speaker_df["wer"].mean(),

            "median_wer":
                speaker_df["wer"].median(),

            "mean_cer":
                speaker_df["cer"].mean(),

            "median_cer":
                speaker_df["cer"].median(),

            "corpus_wer":
                corpus_metrics[
                    "corpus_wer"
                ],

            "corpus_cer":
                corpus_metrics[
                    "corpus_cer"
                ],
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CLASS-LEVEL CORPUS METRICS
# ============================================================

def calculate_class_corpus_metrics(data):

    rows = []

    for speech_status, group in data.groupby(
        "speech_status"
    ):

        metrics = (
            calculate_corpus_metrics(
                group
            )
        )

        rows.append(
            {
                "speech_status":
                    speech_status,

                "recordings":
                    len(group),

                "corpus_wer":
                    metrics[
                        "corpus_wer"
                    ],

                "corpus_cer":
                    metrics[
                        "corpus_cer"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SPEAKER-BALANCED SUMMARY
# ============================================================

def calculate_speaker_balanced_summary(
    speaker_results,
):
    """
    Average speaker-level corpus metrics.

    Every speaker contributes equally regardless
    of how many recordings they have.
    """

    return (
        speaker_results
        .groupby("speech_status")
        .agg(
            speakers=(
                "speaker_id",
                "count",
            ),
            mean_speaker_corpus_wer=(
                "corpus_wer",
                "mean",
            ),
            std_speaker_corpus_wer=(
                "corpus_wer",
                "std",
            ),
            mean_speaker_corpus_cer=(
                "corpus_cer",
                "mean",
            ),
            std_speaker_corpus_cer=(
                "corpus_cer",
                "std",
            ),
        )
        .reset_index()
    )


# ============================================================
# WORST CASES
# ============================================================

def find_worst_cases(
    data,
    n=N_WORST_CASES,
):
    """
    Find recordings with the largest WER.

    CER is used as a secondary sorting criterion.
    """

    columns = [
        "dataset_index",
        "speaker_id",
        "speech_status",
        "gender",
        "reference",
        "prediction",
        "wer",
        "cer",
    ]

    worst = (
        data
        .sort_values(
            by=[
                "wer",
                "cer",
            ],
            ascending=False,
        )
        .head(n)
        [columns]
        .copy()
    )

    return worst


# ============================================================
# PERFECT TRANSCRIPTIONS
# ============================================================

def calculate_perfect_transcriptions(data):

    perfect = (
        data["wer"] == 0
    )

    result = (
        data
        .assign(
            perfect_transcription=perfect
        )
        .groupby(
            "speech_status"
        )
        .agg(
            recordings=(
                "dataset_index",
                "count",
            ),
            perfect_recordings=(
                "perfect_transcription",
                "sum",
            ),
            perfect_rate=(
                "perfect_transcription",
                "mean",
            ),
        )
        .reset_index()
    )

    return result


# ============================================================
# REFERENCE LENGTH ANALYSIS
# ============================================================

def add_reference_lengths(data):
    """
    Add reference word and character counts.

    This helps us understand whether extreme WER/CER
    values occur mostly for very short references.
    """

    data = data.copy()

    data[
        "reference_word_count"
    ] = (
        data["reference_normalized"]
        .fillna("")
        .str.split()
        .str.len()
    )

    data[
        "reference_character_count"
    ] = (
        data["reference_normalized"]
        .fillna("")
        .str.len()
    )

    return data


# ============================================================
# PRINT WORST CASES
# ============================================================

def print_worst_cases(worst_cases):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WORST WHISPER PREDICTIONS"
    )

    print(
        "=" * 80
    )

    for rank, (_, row) in enumerate(
        worst_cases.iterrows(),
        start=1,
    ):

        print(
            f"\n#{rank}"
        )

        print(
            f"Dataset index: "
            f"{row['dataset_index']}"
        )

        print(
            f"Speaker: "
            f"{row['speaker_id']}"
        )

        print(
            f"Status: "
            f"{row['speech_status']}"
        )

        print(
            f"Reference:"
        )

        print(
            row["reference"]
        )

        print(
            "\nWhisper:"
        )

        print(
            row["prediction"]
        )

        print(
            f"\nWER: "
            f"{row['wer']:.3f}"
        )

        print(
            f"CER: "
            f"{row['cer']:.3f}"
        )

        print(
            "-" * 80
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading Whisper predictions..."
    )

    data = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        f"Recordings: {len(data)}"
    )

    print(
        f"Speakers:   "
        f"{data['speaker_id'].nunique()}"
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    required_columns = {
        "dataset_index",
        "speaker_id",
        "speech_status",
        "reference",
        "prediction",
        "reference_normalized",
        "prediction_normalized",
        "wer",
        "cer",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


    # --------------------------------------------------------
    # REMOVE FAILED RECORDINGS
    # --------------------------------------------------------

    valid = data[
        data["wer"].notna()
        & data["cer"].notna()
    ].copy()

    print(
        f"Valid predictions: "
        f"{len(valid)}"
    )


    # --------------------------------------------------------
    # ADD REFERENCE LENGTHS
    # --------------------------------------------------------

    valid = add_reference_lengths(
        valid
    )


    # ========================================================
    # UTTERANCE-LEVEL RESULTS
    # ========================================================

    utterance_summary = (
        calculate_utterance_summary(
            valid
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "UTTERANCE-LEVEL RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        utterance_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # ========================================================
    # CLASS CORPUS METRICS
    # ========================================================

    class_corpus = (
        calculate_class_corpus_metrics(
            valid
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CORPUS-LEVEL RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        class_corpus.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # ========================================================
    # SPEAKER RESULTS
    # ========================================================

    speaker_results = (
        calculate_speaker_metrics(
            valid
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SPEAKER-LEVEL RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        speaker_results.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # ========================================================
    # SPEAKER-BALANCED SUMMARY
    # ========================================================

    speaker_balanced = (
        calculate_speaker_balanced_summary(
            speaker_results
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SPEAKER-BALANCED SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        speaker_balanced.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # ========================================================
    # PERFECT TRANSCRIPTIONS
    # ========================================================

    perfect_results = (
        calculate_perfect_transcriptions(
            valid
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PERFECT TRANSCRIPTIONS"
    )

    print(
        "=" * 80
    )

    print(
        perfect_results.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # ========================================================
    # WORST CASES
    # ========================================================

    worst_cases = (
        find_worst_cases(
            valid
        )
    )

    print_worst_cases(
        worst_cases
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    speaker_results.to_csv(
        SPEAKER_RESULTS_PATH,
        index=False,
        encoding="utf-8",
    )

    worst_cases.to_csv(
        WORST_CASES_PATH,
        index=False,
        encoding="utf-8",
    )


    # ========================================================
    # REPORT
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "NEUROSPEECH WHISPER "
            "SUBSET ANALYSIS\n"
        )

        file.write(
            "=" * 80
            + "\n\n"
        )

        file.write(
            f"Recordings: "
            f"{len(valid)}\n"
        )

        file.write(
            f"Speakers: "
            f"{valid['speaker_id'].nunique()}"
            f"\n\n"
        )


        # ----------------------------------------------------
        # UTTERANCE
        # ----------------------------------------------------

        file.write(
            "UTTERANCE-LEVEL RESULTS\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            utterance_summary.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write(
            "\n\n"
        )


        # ----------------------------------------------------
        # CORPUS
        # ----------------------------------------------------

        file.write(
            "CORPUS-LEVEL RESULTS\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            class_corpus.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write(
            "\n\n"
        )


        # ----------------------------------------------------
        # SPEAKER
        # ----------------------------------------------------

        file.write(
            "SPEAKER-LEVEL RESULTS\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            speaker_results.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write(
            "\n\n"
        )


        # ----------------------------------------------------
        # SPEAKER BALANCED
        # ----------------------------------------------------

        file.write(
            "SPEAKER-BALANCED SUMMARY\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            speaker_balanced.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write(
            "\n\n"
        )


        # ----------------------------------------------------
        # PERFECT
        # ----------------------------------------------------

        file.write(
            "PERFECT TRANSCRIPTIONS\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        file.write(
            perfect_results.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write(
            "\n\n"
        )


        # ----------------------------------------------------
        # WORST CASES
        # ----------------------------------------------------

        file.write(
            "WORST CASES\n"
        )

        file.write(
            "-" * 80
            + "\n"
        )

        for rank, (_, row) in enumerate(
            worst_cases.iterrows(),
            start=1,
        ):

            file.write(
                f"\n#{rank}\n"
            )

            file.write(
                f"Speaker: "
                f"{row['speaker_id']}\n"
            )

            file.write(
                f"Status: "
                f"{row['speech_status']}\n"
            )

            file.write(
                f"Reference: "
                f"{row['reference']}\n"
            )

            file.write(
                f"Whisper: "
                f"{row['prediction']}\n"
            )

            file.write(
                f"WER: "
                f"{row['wer']:.3f}\n"
            )

            file.write(
                f"CER: "
                f"{row['cer']:.3f}\n"
            )


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\nAnalysis saved to:"
    )

    print(
        REPORT_PATH
    )

    print(
        "\nSpeaker results saved to:"
    )

    print(
        SPEAKER_RESULTS_PATH
    )

    print(
        "\nWorst cases saved to:"
    )

    print(
        WORST_CASES_PATH
    )


if __name__ == "__main__":
    main()