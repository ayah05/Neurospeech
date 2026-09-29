from pathlib import Path

import pandas as pd
from datasets import load_dataset, Audio


DATASET_NAME = "abnerh/TORGO-database"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def extract_filename(audio_metadata):
    """
    Extract the filename from the Hugging Face audio metadata.

    Example:
        FC01_1_arrayMic_0066.wav
    """
    path = audio_metadata["path"]

    if path is None:
        raise ValueError("Audio sample does not contain a path.")

    return Path(path).name


def extract_speaker_id(filename):
    """
    Extract the TORGO speaker ID from the filename.

    Examples:
        FC01_1_arrayMic_0066.wav -> FC01
        M05_2_headMic_0219.wav   -> M05
    """
    return filename.split("_")[0]


def main():
    print("Loading TORGO...")

    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]

    # We only need the filename, not the decoded waveform.
    metadata_dataset = train.cast_column(
        "audio",
        Audio(decode=False)
    )

    print(f"Number of recordings: {len(metadata_dataset)}")

    rows = []

    # ---------------------------------------------------------
    # BUILD METADATA TABLE
    # ---------------------------------------------------------

    print("\nBuilding metadata table...")

    for index, sample in enumerate(metadata_dataset):

        filename = extract_filename(sample["audio"])
        speaker_id = extract_speaker_id(filename)

        rows.append(
            {
                "dataset_index": index,
                "speaker_id": speaker_id,
                "speech_status": sample["speech_status"],
                "gender": sample["gender"],
                "transcription": sample["transcription"].strip(),
                "duration": sample["duration"],
                "filename": filename,
            }
        )

    metadata_df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\nValidating metadata...")

    # Every original recording should appear exactly once
    if len(metadata_df) != len(metadata_dataset):
        raise ValueError(
            "Number of metadata rows does not match dataset size."
        )

    # Dataset indices should be unique
    if not metadata_df["dataset_index"].is_unique:
        raise ValueError("Duplicate dataset indices found.")

    # Check missing speaker IDs
    if metadata_df["speaker_id"].isna().any():
        raise ValueError("Missing speaker IDs found.")

    # Check missing filenames
    if metadata_df["filename"].isna().any():
        raise ValueError("Missing filenames found.")

    # A speaker should have exactly one speech status
    status_per_speaker = (
        metadata_df
        .groupby("speaker_id")["speech_status"]
        .nunique()
    )

    if (status_per_speaker > 1).any():
        raise ValueError(
            "At least one speaker has multiple speech-status labels."
        )

    # A speaker should have exactly one gender
    gender_per_speaker = (
        metadata_df
        .groupby("speaker_id")["gender"]
        .nunique()
    )

    if (gender_per_speaker > 1).any():
        raise ValueError(
            "At least one speaker has multiple gender labels."
        )

    print("✓ Number of rows is correct")
    print("✓ Dataset indices are unique")
    print("✓ Speaker IDs are available")
    print("✓ Filenames are available")
    print("✓ Every speaker has exactly one speech status")
    print("✓ Every speaker has exactly one gender")

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    DATA_DIR.mkdir(exist_ok=True)

    output_path = DATA_DIR / "metadata.csv"

    metadata_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("METADATA SUMMARY")
    print("=" * 70)

    print(f"Recordings:      {len(metadata_df)}")
    print(f"Speakers:        {metadata_df['speaker_id'].nunique()}")
    print(f"Transcriptions:  {metadata_df['transcription'].nunique()}")

    print("\nFirst 10 rows:")

    print(
        metadata_df.head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("METADATA CREATED")
    print("=" * 70)

    print(f"Saved to:\n{output_path}")


if __name__ == "__main__":
    main()