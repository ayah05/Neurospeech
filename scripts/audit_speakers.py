from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Audio


DATASET_NAME = "abnerh/TORGO-database"

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def extract_speaker_id(audio_metadata):
    """
    Extract the speaker ID from a TORGO filename.

    Example:
        FC01_1_arrayMic_0066.wav -> FC01
        M05_2_headMic_0219.wav   -> M05
    """

    path = audio_metadata["path"]
    filename = Path(path).name

    return filename.split("_")[0]


def main():
    print("Loading TORGO...")

    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]

    # We only need the filename here, not the decoded waveform.
    metadata_dataset = train.cast_column(
        "audio",
        Audio(decode=False)
    )

    print(f"Number of recordings: {len(metadata_dataset)}")

    # ---------------------------------------------------------
    # COLLECT SPEAKER INFORMATION
    # ---------------------------------------------------------

    rows = []

    for sample in metadata_dataset:

        speaker_id = extract_speaker_id(sample["audio"])

        rows.append(
            {
                "speaker_id": speaker_id,
                "speech_status": sample["speech_status"],
                "gender": sample["gender"],
                "duration": sample["duration"],
            }
        )

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # SPEAKER SUMMARY
    # ---------------------------------------------------------

    speaker_df = (
        df.groupby("speaker_id")
        .agg(
            speech_status=("speech_status", "first"),
            gender=("gender", "first"),
            recordings=("speaker_id", "size"),
            total_duration_seconds=("duration", "sum"),
            mean_duration_seconds=("duration", "mean"),
        )
        .reset_index()
    )

    speaker_df["total_duration_minutes"] = (
        speaker_df["total_duration_seconds"] / 60
    )

    speaker_df = speaker_df[
        [
            "speaker_id",
            "speech_status",
            "gender",
            "recordings",
            "total_duration_minutes",
            "mean_duration_seconds",
        ]
    ]

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPEAKER SUMMARY")
    print("=" * 70)

    print(
        speaker_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\n" + "=" * 70)
    print("NUMBER OF UNIQUE SPEAKERS")
    print("=" * 70)

    print(speaker_df["speaker_id"].nunique())

    print("\nSpeakers by speech status:")

    status_counts = Counter(speaker_df["speech_status"])

    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    print("\nSpeakers by gender:")

    gender_counts = Counter(speaker_df["gender"])

    for gender, count in gender_counts.items():
        print(f"  {gender}: {count}")

    # ---------------------------------------------------------
    # SANITY CHECK
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SANITY CHECK")
    print("=" * 70)

    inconsistent_status = (
        df.groupby("speaker_id")["speech_status"]
        .nunique()
    )

    inconsistent_status = inconsistent_status[
        inconsistent_status > 1
    ]

    inconsistent_gender = (
        df.groupby("speaker_id")["gender"]
        .nunique()
    )

    inconsistent_gender = inconsistent_gender[
        inconsistent_gender > 1
    ]

    if inconsistent_status.empty:
        print("✓ Every speaker has exactly one speech status.")
    else:
        print("WARNING: Speakers with multiple speech-status labels:")
        print(inconsistent_status)

    if inconsistent_gender.empty:
        print("✓ Every speaker has exactly one gender.")
    else:
        print("WARNING: Speakers with multiple gender labels:")
        print(inconsistent_gender)

    # ---------------------------------------------------------
    # SAVE REPORT
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(exist_ok=True)

    output_path = RESULTS_DIR / "torgo_speaker_audit.txt"

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("TORGO SPEAKER AUDIT\n")
        file.write("=" * 70 + "\n\n")

        file.write(
            speaker_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}"
            )
        )

        file.write("\n\n")

        file.write(
            f"Total unique speakers: "
            f"{speaker_df['speaker_id'].nunique()}\n"
        )

        file.write("\nSpeakers by speech status:\n")

        for status, count in status_counts.items():
            file.write(f"{status}: {count}\n")

        file.write("\nSpeakers by gender:\n")

        for gender, count in gender_counts.items():
            file.write(f"{gender}: {count}\n")

    print(f"\nSpeaker audit saved to:\n{output_path}")


if __name__ == "__main__":
    main()