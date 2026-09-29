from collections import Counter
from datasets import load_dataset, Audio
import numpy as np


DATASET_NAME = "abnerh/TORGO-database"


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    print("Loading TORGO...")

    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]

    # We don't need to decode thousands of audio files for the audit.
    # Metadata is sufficient and makes the script much faster.
    train = train.cast_column("audio", Audio(decode=True))

    # ---------------------------------------------------------
    # 1. BASIC DATASET INFORMATION
    # ---------------------------------------------------------

    print_section("DATASET OVERVIEW")

    print(f"Number of samples: {len(train)}")
    print(f"Columns: {train.column_names}")

    # ---------------------------------------------------------
    # 2. CLASS DISTRIBUTION
    # ---------------------------------------------------------

    print_section("SPEECH STATUS DISTRIBUTION")

    speech_status = train["speech_status"]
    status_counts = Counter(speech_status)

    for status, count in status_counts.items():
        percentage = count / len(train) * 100

        print(
            f"{status}: {count} samples "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # 3. GENDER DISTRIBUTION
    # ---------------------------------------------------------

    print_section("GENDER DISTRIBUTION")

    genders = train["gender"]
    gender_counts = Counter(genders)

    for gender, count in gender_counts.items():
        percentage = count / len(train) * 100

        print(
            f"{gender}: {count} samples "
            f"({percentage:.2f}%)"
        )

    # ---------------------------------------------------------
    # 4. SPEECH STATUS × GENDER
    # ---------------------------------------------------------

    print_section("SPEECH STATUS × GENDER")

    combination_counts = Counter(
        zip(
            train["speech_status"],
            train["gender"]
        )
    )

    for (status, gender), count in sorted(combination_counts.items()):
        print(f"{status:12} | {gender:8} | {count}")

    # ---------------------------------------------------------
    # 5. AUDIO DURATION
    # ---------------------------------------------------------

    print_section("AUDIO DURATION")

    durations = np.array(train["duration"], dtype=float)

    print(f"Total duration: {durations.sum() / 3600:.2f} hours")
    print(f"Mean duration:  {durations.mean():.2f} seconds")
    print(f"Median duration: {np.median(durations):.2f} seconds")
    print(f"Min duration:    {durations.min():.2f} seconds")
    print(f"Max duration:    {durations.max():.2f} seconds")

    print("\nDuration percentiles:")

    for percentile in [25, 50, 75, 90, 95, 99]:
        value = np.percentile(durations, percentile)

        print(
            f"{percentile:>2}th percentile: "
            f"{value:.2f} seconds"
        )

    # ---------------------------------------------------------
    # 6. DURATION BY SPEECH STATUS
    # ---------------------------------------------------------

    print_section("DURATION BY SPEECH STATUS")

    for status in sorted(status_counts.keys()):

        status_durations = np.array(
            [
                duration
                for duration, sample_status
                in zip(train["duration"], train["speech_status"])
                if sample_status == status
            ],
            dtype=float
        )

        print(f"\n{status}")
        print(f"  Samples:        {len(status_durations)}")
        print(f"  Total duration: {status_durations.sum() / 3600:.2f} hours")
        print(f"  Mean duration:  {status_durations.mean():.2f} seconds")
        print(f"  Median duration:{np.median(status_durations):.2f} seconds")

    # ---------------------------------------------------------
    # 7. TRANSCRIPTION QUALITY
    # ---------------------------------------------------------

    print_section("TRANSCRIPTION AUDIT")

    transcriptions = train["transcription"]

    missing_transcriptions = sum(
        transcription is None
        or not str(transcription).strip()
        for transcription in transcriptions
    )

    print(f"Missing/empty transcriptions: {missing_transcriptions}")

    normalized_transcriptions = [
        str(transcription).strip().lower()
        for transcription in transcriptions
        if transcription is not None
        and str(transcription).strip()
    ]

    unique_transcriptions = set(normalized_transcriptions)

    print(f"Unique transcriptions: {len(unique_transcriptions)}")
    print(
        "Repeated transcription samples: "
        f"{len(normalized_transcriptions) - len(unique_transcriptions)}"
    )

    transcription_counts = Counter(normalized_transcriptions)

    print("\nMost common transcriptions:")

    for transcription, count in transcription_counts.most_common(10):
        print(f"{count:>4} × {repr(transcription)}")

    # ---------------------------------------------------------
    # 8. AUDIO METADATA
    # ---------------------------------------------------------

    print_section("AUDIO METADATA")

    print("Inspecting 5 healthy and 5 dysarthria audio entries...\n")

    healthy_samples = []
    dysarthria_samples = []

    for i, sample in enumerate(train):

        if sample["speech_status"] == "healthy" and len(healthy_samples) < 5:
            healthy_samples.append((i, sample))

        elif sample["speech_status"] == "dysarthria" and len(dysarthria_samples) < 5:
            dysarthria_samples.append((i, sample))

        # Stop once we have enough of both
        if len(healthy_samples) == 5 and len(dysarthria_samples) == 5:
            break

    print("--- HEALTHY SAMPLES ---\n")

    for index, sample in healthy_samples:
        print(f"Sample {index}")
        print(f"  Audio:         {sample['audio']}")
        print(f"  Transcription: {sample['transcription']}")
        print(f"  Status:        {sample['speech_status']}")
        print(f"  Gender:        {sample['gender']}")
        print(f"  Duration:      {sample['duration']}")
        print()

    print("--- DYSARTHRIA SAMPLES ---\n")

    for index, sample in dysarthria_samples:
        print(f"Sample {index}")
        print(f"  Audio:         {sample['audio']}")
        print(f"  Transcription: {sample['transcription']}")
        print(f"  Status:        {sample['speech_status']}")
        print(f"  Gender:        {sample['gender']}")
        print(f"  Duration:      {sample['duration']}")
        print()

    # ---------------------------------------------------------
    # 9. POSSIBLE SPEAKER INFORMATION
    # ---------------------------------------------------------

    print_section("SPEAKER-ID INVESTIGATION")

    first_audio = train[0]["audio"]

    print("Raw audio metadata:")
    print(first_audio)

    if isinstance(first_audio, dict):

        print("\nAvailable audio metadata fields:")

        for key, value in first_audio.items():
            print(f"  {key}: {value}")

        if "path" in first_audio:
            print("\nAudio path:")
            print(first_audio["path"])

            print(
                "\nCheck whether the filename/path contains "
                "a TORGO speaker identifier."
            )

    else:
        print(
            "\nNo dictionary-style audio metadata available. "
            "Speaker IDs cannot be inferred from this field directly."
        )

    # ---------------------------------------------------------
    # 10. CREATE DATAFRAMES
    # ---------------------------------------------------------

    print_section("CREATING DATAFRAMES")

    import pandas as pd

    # Speech status distribution
    status_df = pd.DataFrame(
        [
            {
                "Speech Status": status,
                "Samples": count,
                "Percentage": round(count / len(train) * 100, 2)
            }
            for status, count in status_counts.items()
        ]
    )

    # Gender distribution
    gender_df = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Samples": count,
                "Percentage": round(count / len(train) * 100, 2)
            }
            for gender, count in gender_counts.items()
        ]
    )

    # Speech status × gender
    status_gender_df = pd.DataFrame(
        [
            {
                "Speech Status": status,
                "Gender": gender,
                "Samples": count
            }
            for (status, gender), count
            in sorted(combination_counts.items())
        ]
    )

    # Overall duration statistics
    duration_df = pd.DataFrame(
        [
            {
                "Samples": len(train),
                "Total Hours": round(durations.sum() / 3600, 2),
                "Mean Seconds": round(durations.mean(), 2),
                "Median Seconds": round(np.median(durations), 2),
                "Min Seconds": round(durations.min(), 2),
                "Max Seconds": round(durations.max(), 2)
            }
        ]
    )

    # Selected audio samples
    sample_rows = []

    for index, sample in healthy_samples + dysarthria_samples:
        sample_rows.append(
            {
                "Dataset Index": index,
                "Transcription": sample["transcription"],
                "Speech Status": sample["speech_status"],
                "Gender": sample["gender"],
                "Duration": sample["duration"],
                "Audio": str(sample["audio"])
            }
        )

    samples_df = pd.DataFrame(sample_rows)

    # ---------------------------------------------------------
    # 11. SAVE AUDIT REPORT
    # ---------------------------------------------------------

    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    report_path = results_dir / "torgo_audit.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("TORGO DATASET AUDIT\n")
        file.write("=" * 80 + "\n\n")

        # Dataset overview
        file.write("DATASET OVERVIEW\n")
        file.write("-" * 80 + "\n")
        file.write(f"Number of samples: {len(train)}\n")
        file.write(f"Columns: {train.column_names}\n\n")

        # Speech status
        file.write("SPEECH STATUS DISTRIBUTION\n")
        file.write("-" * 80 + "\n")
        file.write(status_df.to_string(index=False))
        file.write("\n\n")

        # Gender
        file.write("GENDER DISTRIBUTION\n")
        file.write("-" * 80 + "\n")
        file.write(gender_df.to_string(index=False))
        file.write("\n\n")

        # Speech status x gender
        file.write("SPEECH STATUS × GENDER\n")
        file.write("-" * 80 + "\n")
        file.write(status_gender_df.to_string(index=False))
        file.write("\n\n")

        # Duration
        file.write("AUDIO DURATION\n")
        file.write("-" * 80 + "\n")
        file.write(duration_df.to_string(index=False))
        file.write("\n\n")

        # Transcriptions
        file.write("TRANSCRIPTION AUDIT\n")
        file.write("-" * 80 + "\n")
        file.write(
            f"Missing/empty transcriptions: {missing_transcriptions}\n"
        )
        file.write(
            f"Unique transcriptions: {len(unique_transcriptions)}\n"
        )
        file.write(
            "Repeated transcription samples: "
            f"{len(normalized_transcriptions) - len(unique_transcriptions)}\n\n"
        )

        # Example samples
        file.write("SELECTED AUDIO SAMPLES\n")
        file.write("-" * 80 + "\n")
        file.write(samples_df.to_string(index=False))
        file.write("\n")

    print(f"\nAudit report saved to:")
    print(report_path)

    # ---------------------------------------------------------
    # SPEAKER-ID / FILE PATH INVESTIGATION
    # ---------------------------------------------------------

    print_section("AUDIO FILE PATH INVESTIGATION")

    # Create a metadata-only view.
    # This does NOT modify the normal `train` dataset.
    train_metadata = train.cast_column(
        "audio",
        Audio(decode=False)
    )

    indices_to_check = [
        healthy_samples[0][0],
        dysarthria_samples[0][0]
    ]

    for index in indices_to_check:

        sample = train_metadata[index]
        audio_metadata = sample["audio"]

        print(f"\nSample {index}")
        print(f"  Status:        {sample['speech_status']}")
        print(f"  Gender:        {sample['gender']}")
        print(f"  Transcription: {sample['transcription']}")

        # IMPORTANT:
        # Do not print audio_metadata itself because it may contain
        # the entire WAV file as bytes.

        if isinstance(audio_metadata, dict):
            print(f"  Audio path:    {audio_metadata.get('path')}")
            print(f"  Metadata keys: {list(audio_metadata.keys())}")
        else:
            print(f"  Metadata type: {type(audio_metadata)}")

    print_section("AUDIT COMPLETE")



if __name__ == "__main__":
    main()