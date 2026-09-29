from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = ROOT / "data" / "metadata.csv"
OUTPUT_PATH = ROOT / "data" / "metadata_with_folds.csv"
REPORT_PATH = ROOT / "results" / "fold_summary.txt"

N_FOLDS = 5
SPEAKERS_PER_FOLD = 3


def build_speaker_table(metadata):
    """
    Create one row per speaker.
    """

    speaker_df = (
        metadata
        .groupby("speaker_id")
        .agg(
            speech_status=("speech_status", "first"),
            gender=("gender", "first"),
            recordings=("speaker_id", "size"),
            total_duration=("duration", "sum"),
        )
        .reset_index()
    )

    return speaker_df


def validate_speaker_table(speaker_df):
    """
    Basic checks before creating folds.
    """

    if len(speaker_df) != 15:
        raise ValueError(
            f"Expected 15 speakers, found {len(speaker_df)}."
        )

    status_counts = speaker_df["speech_status"].value_counts()

    if status_counts.get("dysarthria", 0) != 8:
        raise ValueError("Expected 8 dysarthria speakers.")

    if status_counts.get("healthy", 0) != 7:
        raise ValueError("Expected 7 healthy speakers.")

    print("✓ 15 speakers found")
    print("✓ 8 dysarthria speakers")
    print("✓ 7 healthy speakers")


def fold_score(fold, target_recordings, target_duration):
    """
    Lower score = better fold.

    The score considers:
    - recording-count balance
    - duration balance
    - gender balance
    """

    recordings = fold["recordings"].sum()
    duration = fold["total_duration"].sum()

    recording_difference = abs(
        recordings - target_recordings
    ) / target_recordings

    duration_difference = abs(
        duration - target_duration
    ) / target_duration

    female_count = (fold["gender"] == "female").sum()
    male_count = (fold["gender"] == "male").sum()

    gender_difference = abs(female_count - male_count)

    return (
        recording_difference
        + duration_difference
        + 0.25 * gender_difference
    )


def create_candidate_folds(
    speaker_df,
    dysarthria_count,
    healthy_count,
    target_recordings,
    target_duration,
):
    """
    Generate all possible 3-speaker folds having the requested
    class composition.
    """

    dysarthria = speaker_df[
        speaker_df["speech_status"] == "dysarthria"
    ]

    healthy = speaker_df[
        speaker_df["speech_status"] == "healthy"
    ]

    candidates = []

    for dys_indices in combinations(
        dysarthria.index,
        dysarthria_count
    ):

        for healthy_indices in combinations(
            healthy.index,
            healthy_count
        ):

            indices = list(dys_indices) + list(healthy_indices)

            fold = speaker_df.loc[indices]

            score = fold_score(
                fold,
                target_recordings,
                target_duration
            )

            candidates.append(
                {
                    "speakers": tuple(
                        sorted(fold["speaker_id"].tolist())
                    ),
                    "score": score,
                }
            )

    return candidates


def find_best_assignment(
    candidate_sets,
    fold_number=0,
    used_speakers=None,
    current_assignment=None,
    current_score=0,
    best=None,
):
    """
    Search recursively for the lowest-scoring set of folds
    where every speaker appears exactly once.
    """

    if used_speakers is None:
        used_speakers = set()

    if current_assignment is None:
        current_assignment = []

    if best is None:
        best = {
            "score": float("inf"),
            "assignment": None,
        }

    # All five folds assigned
    if fold_number == len(candidate_sets):

        if current_score < best["score"]:
            best["score"] = current_score
            best["assignment"] = current_assignment.copy()

        return best

    # No reason to continue if already worse
    if current_score >= best["score"]:
        return best

    for candidate in candidate_sets[fold_number]:

        speakers = set(candidate["speakers"])

        # Speaker already used in another fold
        if speakers & used_speakers:
            continue

        best = find_best_assignment(
            candidate_sets=candidate_sets,
            fold_number=fold_number + 1,
            used_speakers=used_speakers | speakers,
            current_assignment=(
                current_assignment + [candidate]
            ),
            current_score=current_score + candidate["score"],
            best=best,
        )

    return best


def create_fold_summary(metadata):
    """
    Create one summary row per fold.
    """

    rows = []

    for fold_number in sorted(metadata["fold"].unique()):

        fold = metadata[
            metadata["fold"] == fold_number
        ]

        speaker_info = (
            fold[
                [
                    "speaker_id",
                    "speech_status",
                    "gender"
                ]
            ]
            .drop_duplicates()
        )

        rows.append(
            {
                "fold": fold_number,
                "speakers": speaker_info["speaker_id"].nunique(),
                "healthy": (
                    speaker_info["speech_status"] == "healthy"
                ).sum(),
                "dysarthria": (
                    speaker_info["speech_status"] == "dysarthria"
                ).sum(),
                "female": (
                    speaker_info["gender"] == "female"
                ).sum(),
                "male": (
                    speaker_info["gender"] == "male"
                ).sum(),
                "recordings": len(fold),
                "duration_hours": fold["duration"].sum() / 3600,
            }
        )

    return pd.DataFrame(rows)


def main():

    print("Loading metadata...")

    metadata = pd.read_csv(METADATA_PATH)

    # ---------------------------------------------------------
    # SPEAKER TABLE
    # ---------------------------------------------------------

    speaker_df = build_speaker_table(metadata)

    validate_speaker_table(speaker_df)

    print("\nSpeaker table:")
    print(speaker_df.to_string(index=False))

    # ---------------------------------------------------------
    # TARGET FOLD SIZE
    # ---------------------------------------------------------

    target_recordings = (
        speaker_df["recordings"].sum() / N_FOLDS
    )

    target_duration = (
        speaker_df["total_duration"].sum() / N_FOLDS
    )

    print("\nTarget per fold:")
    print(
        f"  Recordings: {target_recordings:.1f}"
    )
    print(
        f"  Duration:   {target_duration / 3600:.2f} hours"
    )

    # ---------------------------------------------------------
    # CLASS COMPOSITION
    # ---------------------------------------------------------

    # We have:
    #
    # 8 dysarthria
    # 7 healthy
    #
    # Therefore:
    #
    # 3 folds -> 2 dysarthria + 1 healthy
    # 2 folds -> 1 dysarthria + 2 healthy

    fold_compositions = [
        (2, 1),
        (2, 1),
        (2, 1),
        (1, 2),
        (1, 2),
    ]

    candidate_sets = []

    for dys_count, healthy_count in fold_compositions:

        candidates = create_candidate_folds(
            speaker_df=speaker_df,
            dysarthria_count=dys_count,
            healthy_count=healthy_count,
            target_recordings=target_recordings,
            target_duration=target_duration,
        )

        # Try better candidates first
        candidates.sort(
            key=lambda candidate: candidate["score"]
        )

        candidate_sets.append(candidates)

    # ---------------------------------------------------------
    # FIND BEST ASSIGNMENT
    # ---------------------------------------------------------

    print("\nSearching for balanced fold assignment...")

    best = find_best_assignment(candidate_sets)

    if best["assignment"] is None:
        raise RuntimeError(
            "Could not find a valid fold assignment."
        )

    print(
        f"Best balance score: {best['score']:.4f}"
    )

    # ---------------------------------------------------------
    # ASSIGN FOLD NUMBERS
    # ---------------------------------------------------------

    speaker_to_fold = {}

    for fold_number, candidate in enumerate(
        best["assignment"],
        start=1
    ):

        for speaker in candidate["speakers"]:
            speaker_to_fold[speaker] = fold_number

    metadata["fold"] = metadata["speaker_id"].map(
        speaker_to_fold
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\nValidating folds...")

    if metadata["fold"].isna().any():
        raise ValueError(
            "At least one recording has no fold."
        )

    folds_per_speaker = (
        metadata
        .groupby("speaker_id")["fold"]
        .nunique()
    )

    if not (folds_per_speaker == 1).all():
        raise ValueError(
            "Speaker leakage detected."
        )

    speakers_per_fold = (
        metadata[
            ["speaker_id", "fold"]
        ]
        .drop_duplicates()
        .groupby("fold")
        .size()
    )

    if not (
        speakers_per_fold == SPEAKERS_PER_FOLD
    ).all():
        raise ValueError(
            "A fold does not contain exactly 3 speakers."
        )

    print("✓ Every recording has a fold")
    print("✓ Every speaker occurs in exactly one fold")
    print("✓ Every fold contains exactly 3 speakers")
    print("✓ No speaker leakage")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    summary_df = create_fold_summary(metadata)

    print("\n" + "=" * 90)
    print("FOLD SUMMARY")
    print("=" * 90)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\nSpeaker assignments:")

    speaker_assignment_df = (
        metadata[
            [
                "speaker_id",
                "speech_status",
                "gender",
                "fold",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["fold", "speech_status", "speaker_id"]
        )
    )

    print(
        speaker_assignment_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    metadata.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("TORGO 5-FOLD SPEAKER-INDEPENDENT SPLIT\n")
        file.write("=" * 90 + "\n\n")

        file.write("FOLD SUMMARY\n")
        file.write("-" * 90 + "\n")

        file.write(
            summary_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.2f}"
            )
        )

        file.write("\n\nSPEAKER ASSIGNMENTS\n")
        file.write("-" * 90 + "\n")

        file.write(
            speaker_assignment_df.to_string(
                index=False
            )
        )

        file.write("\n")

    print("\nSaved metadata with folds to:")
    print(OUTPUT_PATH)

    print("\nSaved fold report to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()