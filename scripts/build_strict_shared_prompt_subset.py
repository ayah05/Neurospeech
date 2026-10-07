from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT / "data" / "matched_perfect_asr_transcripts.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "strict_shared_prompt_subset.csv"
)

PROMPT_SUMMARY_PATH = (
    ROOT / "results" / "strict_shared_prompt_summary.csv"
)

SUMMARY_PATH = (
    ROOT / "results" / "strict_shared_prompt_summary.txt"
)

MIN_SPEAKERS_PER_GROUP = 3


# ============================================================
# LOAD DATA
# ============================================================

print("Loading matched Perfect-ASR dataset...")

data = pd.read_csv(INPUT_PATH)

print(f"Input recordings: {len(data)}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "dataset_index",
    "speaker_id",
    "speech_status",
    "gender",
    "fold",
    "reference_normalized",
    "prediction_normalized",
    "wer",
}

missing_columns = required_columns - set(data.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )


if not data["dataset_index"].is_unique:
    raise ValueError(
        "dataset_index is not unique."
    )


# ============================================================
# VERIFY PERFECT ASR
# ============================================================

non_perfect = data[
    data["wer"] != 0
]

if len(non_perfect) > 0:
    raise ValueError(
        f"Found {len(non_perfect)} recordings with WER != 0."
    )


# ============================================================
# BUILD PROMPT SUMMARY
# ============================================================

prompt_rows = []

for transcript, group in data.groupby(
    "reference_normalized"
):

    healthy = group[
        group["speech_status"] == "healthy"
    ]

    dysarthria = group[
        group["speech_status"] == "dysarthria"
    ]

    healthy_speakers = (
        healthy["speaker_id"].nunique()
    )

    dysarthria_speakers = (
        dysarthria["speaker_id"].nunique()
    )

    prompt_rows.append(
        {
            "reference_normalized": transcript,

            "total_recordings": len(group),

            "healthy_recordings": len(healthy),

            "dysarthria_recordings": len(dysarthria),

            "healthy_speakers": healthy_speakers,

            "dysarthria_speakers": dysarthria_speakers,

            "total_speakers":
                group["speaker_id"].nunique(),
        }
    )


prompt_summary = pd.DataFrame(prompt_rows)


# ============================================================
# SELECT STRICT PROMPTS
# ============================================================
#
# A prompt is retained only when it has:
#
# >= 3 unique healthy speakers
# AND
# >= 3 unique dysarthria speakers
#
# ============================================================

strict_prompt_summary = prompt_summary[
    (
        prompt_summary["healthy_speakers"]
        >= MIN_SPEAKERS_PER_GROUP
    )
    &
    (
        prompt_summary["dysarthria_speakers"]
        >= MIN_SPEAKERS_PER_GROUP
    )
].copy()


strict_prompt_summary = (
    strict_prompt_summary
    .sort_values(
        [
            "total_speakers",
            "total_recordings",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)


strict_prompts = set(
    strict_prompt_summary[
        "reference_normalized"
    ]
)


print(
    "\n========================================"
)
print("STRICT PROMPT SELECTION")
print("========================================")

print(
    f"Minimum speakers per group: "
    f"{MIN_SPEAKERS_PER_GROUP}"
)

print(
    f"Eligible prompts: "
    f"{len(strict_prompts)}"
)


# ============================================================
# BUILD STRICT DATASET
# ============================================================

strict = data[
    data["reference_normalized"].isin(
        strict_prompts
    )
].copy()


strict = (
    strict
    .sort_values(
        [
            "reference_normalized",
            "speech_status",
            "speaker_id",
            "dataset_index",
        ]
    )
    .reset_index(drop=True)
)


print(
    f"Strict subset recordings: "
    f"{len(strict)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = (
    strict["speech_status"]
    .value_counts()
)


print(
    "\nClass distribution:"
)

print(
    class_counts.to_string()
)


# ============================================================
# SPEAKER DISTRIBUTION
# ============================================================

speaker_summary = (
    strict
    .groupby(
        [
            "speaker_id",
            "speech_status",
            "fold",
        ]
    )
    .size()
    .reset_index(
        name="recordings"
    )
)


print(
    "\n========================================"
)
print("SPEAKER COVERAGE")
print("========================================")

print(
    speaker_summary.to_string(
        index=False
    )
)


total_speakers = (
    strict["speaker_id"].nunique()
)

healthy_speakers = (
    strict.loc[
        strict["speech_status"] == "healthy",
        "speaker_id",
    ]
    .nunique()
)

dysarthria_speakers = (
    strict.loc[
        strict["speech_status"] == "dysarthria",
        "speaker_id",
    ]
    .nunique()
)


print(
    f"\nTotal speakers retained: "
    f"{total_speakers}"
)

print(
    f"Healthy speakers retained: "
    f"{healthy_speakers}"
)

print(
    f"Dysarthria speakers retained: "
    f"{dysarthria_speakers}"
)


# ============================================================
# FOLD DISTRIBUTION
# ============================================================

fold_summary = (
    strict
    .groupby(
        [
            "fold",
            "speech_status",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print(
    "\n========================================"
)
print("FOLD DISTRIBUTION")
print("========================================")

print(
    fold_summary.to_string()
)


# ============================================================
# CHECK EACH FOLD
# ============================================================

print(
    "\n========================================"
)
print("FOLD VALIDATION")
print("========================================"
)


for fold in sorted(
    strict["fold"].unique()
):

    test = strict[
        strict["fold"] == fold
    ]

    train = strict[
        strict["fold"] != fold
    ]


    train_speakers = set(
        train["speaker_id"].unique()
    )

    test_speakers = set(
        test["speaker_id"].unique()
    )

    overlap = (
        train_speakers
        & test_speakers
    )


    healthy_test = (
        test["speech_status"]
        == "healthy"
    ).sum()

    dysarthria_test = (
        test["speech_status"]
        == "dysarthria"
    ).sum()


    print(
        f"\nFold {fold}"
    )

    print(
        f"  Train recordings: "
        f"{len(train)}"
    )

    print(
        f"  Test recordings: "
        f"{len(test)}"
    )

    print(
        f"  Test healthy: "
        f"{healthy_test}"
    )

    print(
        f"  Test dysarthria: "
        f"{dysarthria_test}"
    )

    print(
        f"  Train speakers: "
        f"{len(train_speakers)}"
    )

    print(
        f"  Test speakers: "
        f"{len(test_speakers)}"
    )

    print(
        f"  Speaker overlap: "
        f"{len(overlap)}"
    )


    if overlap:
        raise RuntimeError(
            f"Speaker leakage in fold {fold}: "
            f"{sorted(overlap)}"
        )


    if healthy_test == 0:
        raise RuntimeError(
            f"Fold {fold} contains no "
            f"healthy test recordings."
        )


    if dysarthria_test == 0:
        raise RuntimeError(
            f"Fold {fold} contains no "
            f"dysarthria test recordings."
        )


# ============================================================
# PROMPT EXAMPLES
# ============================================================

print(
    "\n========================================"
)
print("TOP STRICT PROMPTS")
print("========================================"
)


display_columns = [
    "reference_normalized",
    "total_recordings",
    "healthy_recordings",
    "dysarthria_recordings",
    "healthy_speakers",
    "dysarthria_speakers",
    "total_speakers",
]


print(
    strict_prompt_summary[
        display_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# SAVE FILES
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

PROMPT_SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


strict.to_csv(
    OUTPUT_PATH,
    index=False,
)


strict_prompt_summary.to_csv(
    PROMPT_SUMMARY_PATH,
    index=False,
)


# ============================================================
# CREATE TEXT SUMMARY
# ============================================================

summary_lines = []

summary_lines.append(
    "Strict Shared-Prompt Perfect-ASR Dataset"
)

summary_lines.append(
    "=" * 45
)

summary_lines.append("")

summary_lines.append(
    f"Minimum speakers per group: "
    f"{MIN_SPEAKERS_PER_GROUP}"
)

summary_lines.append(
    f"Eligible prompts: "
    f"{len(strict_prompts)}"
)

summary_lines.append(
    f"Total recordings: "
    f"{len(strict)}"
)

summary_lines.append("")

summary_lines.append(
    "Class distribution:"
)

for status, count in class_counts.items():

    summary_lines.append(
        f"  {status}: {count}"
    )


summary_lines.append("")

summary_lines.append(
    f"Total speakers retained: "
    f"{total_speakers}"
)

summary_lines.append(
    f"Healthy speakers retained: "
    f"{healthy_speakers}"
)

summary_lines.append(
    f"Dysarthria speakers retained: "
    f"{dysarthria_speakers}"
)

summary_lines.append("")

summary_lines.append(
    "Recordings per speaker:"
)

summary_lines.append(
    speaker_summary.to_string(
        index=False
    )
)

summary_lines.append("")

summary_lines.append(
    "Fold distribution:"
)

summary_lines.append(
    fold_summary.to_string()
)

summary_lines.append("")

summary_lines.append(
    "Top strict prompts:"
)

summary_lines.append(
    strict_prompt_summary[
        display_columns
    ]
    .head(30)
    .to_string(index=False)
)


summary_text = "\n".join(
    summary_lines
)


SUMMARY_PATH.write_text(
    summary_text,
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n========================================"
)
print("FILES SAVED")
print("========================================"
)

print(OUTPUT_PATH)
print(PROMPT_SUMMARY_PATH)
print(SUMMARY_PATH)