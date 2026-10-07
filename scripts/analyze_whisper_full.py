from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

WHISPER_PATH = (
    ROOT / "data" / "whisper_full_predictions.csv"
)

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "matched_transcripts_summary.txt"
)

PROMPT_RESULTS_PATH = (
    RESULTS_DIR
    / "matched_transcripts_prompt_summary.csv"
)

MATCHED_DATA_PATH = (
    ROOT / "data"
    / "matched_perfect_asr_transcripts.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Whisper predictions...")

data = pd.read_csv(
    WHISPER_PATH
)

print(
    f"Total recordings: {len(data)}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = {
    "dataset_index",
    "speaker_id",
    "speech_status",
    "gender",
    "fold",
    "reference",
    "reference_normalized",
    "prediction",
    "prediction_normalized",
    "wer",
    "cer",
    "error",
}


missing_columns = (
    required_columns
    - set(data.columns)
)


if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )


if not data[
    "dataset_index"
].is_unique:
    raise ValueError(
        "dataset_index is not unique."
    )


# ============================================================
# KEEP SUCCESSFUL WHISPER RECORDINGS
# ============================================================

successful = data[
    data["error"].isna()
].copy()


print(
    f"Successful Whisper recordings: "
    f"{len(successful)}"
)


# ============================================================
# PERFECT-ASR SUBSET
# ============================================================

perfect = successful[
    np.isclose(
        successful["wer"],
        0.0,
    )
].copy()


print(
    f"Perfect-ASR recordings: "
    f"{len(perfect)}"
)


# ============================================================
# REMOVE EMPTY NORMALIZED REFERENCES
# ============================================================

perfect = perfect[
    perfect[
        "reference_normalized"
    ].notna()
].copy()


perfect = perfect[
    perfect[
        "reference_normalized"
    ].str.strip()
    != ""
].copy()


print(
    "Perfect-ASR recordings with "
    f"non-empty reference: {len(perfect)}"
)


# ============================================================
# BASIC PERFECT-ASR STATISTICS
# ============================================================

unique_perfect_transcripts = (
    perfect[
        "reference_normalized"
    ].nunique()
)


perfect_speakers = (
    perfect[
        "speaker_id"
    ].nunique()
)


print(
    f"Unique perfect-ASR transcripts: "
    f"{unique_perfect_transcripts}"
)

print(
    f"Speakers in perfect-ASR subset: "
    f"{perfect_speakers}"
)


# ============================================================
# BUILD PROMPT-LEVEL SUMMARY
# ============================================================
#
# Each row will represent one normalized reference transcript.
#
# We want to know:
#
# - how many recordings exist
# - how many healthy recordings
# - how many dysarthria recordings
# - how many unique healthy speakers
# - how many unique dysarthria speakers
#
# ============================================================

prompt_rows = []


for (
    transcript,
    group,
) in perfect.groupby(
    "reference_normalized"
):

    healthy = group[
        group["speech_status"]
        == "healthy"
    ]

    dysarthria = group[
        group["speech_status"]
        == "dysarthria"
    ]


    healthy_speakers = (
        healthy[
            "speaker_id"
        ].nunique()
    )

    dysarthria_speakers = (
        dysarthria[
            "speaker_id"
        ].nunique()
    )


    prompt_rows.append(
        {
            "reference_normalized":
                transcript,

            "total_recordings":
                len(group),

            "healthy_recordings":
                len(healthy),

            "dysarthria_recordings":
                len(dysarthria),

            "healthy_speakers":
                healthy_speakers,

            "dysarthria_speakers":
                dysarthria_speakers,

            "total_speakers":
                group[
                    "speaker_id"
                ].nunique(),

            "shared_between_groups":
                (
                    healthy_speakers > 0
                    and
                    dysarthria_speakers > 0
                ),
        }
    )


prompt_summary = pd.DataFrame(
    prompt_rows
)


prompt_summary = (
    prompt_summary
    .sort_values(
        [
            "shared_between_groups",
            "total_speakers",
            "total_recordings",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# IDENTIFY SHARED TRANSCRIPTS
# ============================================================
#
# A transcript is eligible when:
#
#   >= 1 healthy speaker
#   AND
#   >= 1 dysarthria speaker
#
# Notice that we use SPEAKERS here rather than simply
# recordings.
#
# ============================================================

shared_prompts = prompt_summary[
    prompt_summary[
        "shared_between_groups"
    ]
].copy()


shared_transcripts = set(
    shared_prompts[
        "reference_normalized"
    ]
)


print(
    "\n========================================"
)

print(
    "SHARED TRANSCRIPTS"
)

print(
    "========================================"
)


print(
    f"Shared transcripts: "
    f"{len(shared_transcripts)}"
)


# ============================================================
# CREATE MATCHED DATASET
# ============================================================

matched = perfect[
    perfect[
        "reference_normalized"
    ].isin(
        shared_transcripts
    )
].copy()


matched = (
    matched
    .sort_values(
        [
            "reference_normalized",
            "speech_status",
            "speaker_id",
            "dataset_index",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# MATCHED DATASET STATISTICS
# ============================================================

matched_class_counts = (
    matched[
        "speech_status"
    ]
    .value_counts()
)


matched_speaker_counts = (
    matched[
        "speaker_id"
    ]
    .nunique()
)


healthy_speakers = (
    matched[
        matched[
            "speech_status"
        ]
        == "healthy"
    ][
        "speaker_id"
    ]
    .nunique()
)


dysarthria_speakers = (
    matched[
        matched[
            "speech_status"
        ]
        == "dysarthria"
    ][
        "speaker_id"
    ]
    .nunique()
)


print(
    f"Matched recordings: "
    f"{len(matched)}"
)


print(
    "\nMatched class distribution:"
)

print(
    matched_class_counts.to_string()
)


print(
    f"\nSpeakers retained: "
    f"{matched_speaker_counts}"
)

print(
    f"Healthy speakers: "
    f"{healthy_speakers}"
)

print(
    f"Dysarthria speakers: "
    f"{dysarthria_speakers}"
)


# ============================================================
# RECORDINGS PER SPEAKER
# ============================================================

speaker_summary = (
    matched
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
    "\nMatched recordings per speaker:"
)

print(
    speaker_summary.to_string(
        index=False
    )
)


# ============================================================
# PROMPT COVERAGE
# ============================================================
#
# More stringent prompt subsets are useful to inspect.
#
# Example:
#
# >= 2 healthy speakers
# >= 2 dysarthria speakers
#
# is considerably stronger than simply having one speaker
# from each group.
#
# ============================================================

shared_1_1 = (
    (
        prompt_summary[
            "healthy_speakers"
        ] >= 1
    )
    &
    (
        prompt_summary[
            "dysarthria_speakers"
        ] >= 1
    )
).sum()


shared_2_2 = (
    (
        prompt_summary[
            "healthy_speakers"
        ] >= 2
    )
    &
    (
        prompt_summary[
            "dysarthria_speakers"
        ] >= 2
    )
).sum()


shared_3_3 = (
    (
        prompt_summary[
            "healthy_speakers"
        ] >= 3
    )
    &
    (
        prompt_summary[
            "dysarthria_speakers"
        ] >= 3
    )
).sum()


print(
    "\nPrompt coverage:"
)

print(
    f">=1 healthy + >=1 dysarthria speaker: "
    f"{shared_1_1}"
)

print(
    f">=2 healthy + >=2 dysarthria speakers: "
    f"{shared_2_2}"
)

print(
    f">=3 healthy + >=3 dysarthria speakers: "
    f"{shared_3_3}"
)


# ============================================================
# TOP SHARED PROMPTS
# ============================================================

top_shared = (
    shared_prompts
    .head(30)
)


print(
    "\n========================================"
)

print(
    "TOP SHARED TRANSCRIPTS"
)

print(
    "========================================"
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
    top_shared[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE DATA
# ============================================================

prompt_summary.to_csv(
    PROMPT_RESULTS_PATH,
    index=False,
)


matched.to_csv(
    MATCHED_DATA_PATH,
    index=False,
)


# ============================================================
# SUMMARY FILE
# ============================================================

summary_lines = []

summary_lines.append(
    "Matched Perfect-ASR Transcript Analysis"
)

summary_lines.append(
    "=" * 45
)

summary_lines.append("")


summary_lines.append(
    f"Total Whisper recordings: "
    f"{len(data)}"
)

summary_lines.append(
    f"Successful Whisper recordings: "
    f"{len(successful)}"
)

summary_lines.append(
    f"Perfect-ASR recordings: "
    f"{len(perfect)}"
)

summary_lines.append(
    f"Unique perfect-ASR transcripts: "
    f"{unique_perfect_transcripts}"
)

summary_lines.append("")


summary_lines.append(
    f"Shared transcripts "
    f"(>=1 healthy + >=1 dysarthria speaker): "
    f"{shared_1_1}"
)

summary_lines.append(
    f"Shared transcripts "
    f"(>=2 healthy + >=2 dysarthria speakers): "
    f"{shared_2_2}"
)

summary_lines.append(
    f"Shared transcripts "
    f"(>=3 healthy + >=3 dysarthria speakers): "
    f"{shared_3_3}"
)

summary_lines.append("")


summary_lines.append(
    f"Matched recordings: "
    f"{len(matched)}"
)


for (
    speech_status,
    count,
) in matched_class_counts.items():

    summary_lines.append(
        f"  {speech_status}: {count}"
    )


summary_lines.append("")


summary_lines.append(
    f"Speakers retained: "
    f"{matched_speaker_counts}"
)

summary_lines.append(
    f"  healthy speakers: "
    f"{healthy_speakers}"
)

summary_lines.append(
    f"  dysarthria speakers: "
    f"{dysarthria_speakers}"
)


summary_lines.append("")

summary_lines.append(
    "Matched recordings per speaker:"
)

summary_lines.append(
    speaker_summary.to_string(
        index=False
    )
)


summary_lines.append("")

summary_lines.append(
    "Top shared transcripts:"
)

summary_lines.append(
    top_shared[
        display_columns
    ].to_string(
        index=False
    )
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

print(
    "FILES SAVED"
)

print(
    "========================================"
)


print(
    MATCHED_DATA_PATH
)

print(
    PROMPT_RESULTS_PATH
)

print(
    SUMMARY_PATH
)