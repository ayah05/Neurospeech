from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FLIPS_PATH = (
    ROOT
    / "results"
    / "prediction_flips_hubert_residual.csv"
)

OUTPUT_DIR = ROOT / "results"

SPEAKER_OUTPUT = (
    OUTPUT_DIR
    / "recording_conditions_by_speaker.csv"
)

WITHIN_SPEAKER_OUTPUT = (
    OUTPUT_DIR
    / "recording_conditions_within_speaker.csv"
)

MIC_OUTPUT = (
    OUTPUT_DIR
    / "recording_conditions_by_microphone.csv"
)


# ============================================================
# CONFIG
# ============================================================

FOCUS_SPEAKERS = [
    "F03",
    "F04",
    "FC01",
    "FC03",
]

ACOUSTIC_FEATURES = [
    "audio_duration",
    "voiced_duration",
    "speech_ratio",
    "pause_ratio",
    "number_of_pauses",
    "mean_pause_duration",
    "f0_mean",
    "f0_std",
    "rms_mean",
    "rms_std",
]


# ============================================================
# HELPERS
# ============================================================

def extract_microphone(filename):

    if pd.isna(filename):
        return "unknown"

    filename = str(filename)

    # Examples such as:
    # F03_..._arrayMic_...
    # F03_..._headMic_...

    match = re.search(
        r"_(arrayMic|headMic)_",
        filename,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).lower()

    return "unknown"


def safe_corr(x, y):

    valid = (
        pd.notna(x)
        & pd.notna(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if x.nunique() < 2:
        return np.nan

    if y.nunique() < 2:
        return np.nan

    return x.corr(y)


# ============================================================
# SPEAKER-LEVEL AUDIT
# ============================================================

def analyze_by_speaker(data):

    rows = []

    for speaker_id, group in data.groupby(
        "speaker_id"
    ):

        row = {
            "speaker_id": speaker_id,
            "speech_status":
                group["speech_status"].iloc[0],
            "fold":
                int(group["fold"].iloc[0]),
            "n_recordings":
                len(group),

            "hubert_accuracy":
                (
                    group["pred_hubert"]
                    == group["true_label"]
                ).mean(),

            "residual_accuracy":
                (
                    group["pred_residual_hubert"]
                    == group["true_label"]
                ).mean(),

            "mean_p_dys_hubert":
                group[
                    "p_dys_hubert"
                ].mean(),

            "mean_p_dys_residual":
                group[
                    "p_dys_residual_hubert"
                ].mean(),

            "mean_delta_p_dys":
                group[
                    "delta_p_dys"
                ].mean(),

            "mean_abs_delta_p_dys":
                group[
                    "abs_delta_p_dys"
                ].mean(),
        }

        for feature in ACOUSTIC_FEATURES:

            if feature not in group.columns:
                continue

            row[
                f"{feature}_mean"
            ] = group[feature].mean()

            row[
                f"{feature}_median"
            ] = group[feature].median()

            row[
                f"{feature}_std"
            ] = group[feature].std()

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# WITHIN-SPEAKER ANALYSIS
# ============================================================

def analyze_within_speaker(data):

    rows = []

    for speaker_id, group in data.groupby(
        "speaker_id"
    ):

        for feature in ACOUSTIC_FEATURES:

            if feature not in group.columns:
                continue

            # --------------------------------------------
            # Does the feature track HuBERT probability?
            # --------------------------------------------

            corr_hubert = safe_corr(
                group[feature],
                group["p_dys_hubert"],
            )

            # --------------------------------------------
            # Does the feature track the probability
            # change caused by residualization?
            # --------------------------------------------

            corr_delta = safe_corr(
                group[feature],
                group["delta_p_dys"],
            )

            # --------------------------------------------
            # Compare recordings strongly affected by
            # residualization with recordings that barely
            # changed.
            # --------------------------------------------

            abs_delta = group[
                "abs_delta_p_dys"
            ]

            q25 = abs_delta.quantile(0.25)
            q75 = abs_delta.quantile(0.75)

            low_change = group[
                abs_delta <= q25
            ]

            high_change = group[
                abs_delta >= q75
            ]

            low_mean = (
                low_change[feature].mean()
            )

            high_mean = (
                high_change[feature].mean()
            )

            row = {
                "speaker_id":
                    speaker_id,

                "speech_status":
                    group[
                        "speech_status"
                    ].iloc[0],

                "fold":
                    int(
                        group["fold"].iloc[0]
                    ),

                "feature":
                    feature,

                "n_recordings":
                    len(group),

                "corr_feature_hubert_p_dys":
                    corr_hubert,

                "corr_feature_delta_p_dys":
                    corr_delta,

                "low_change_mean":
                    low_mean,

                "high_change_mean":
                    high_mean,

                "high_minus_low":
                    high_mean - low_mean,

                "low_change_n":
                    len(low_change),

                "high_change_n":
                    len(high_change),
            }

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MICROPHONE AUDIT
# ============================================================

def analyze_microphone(data):

    if "filename" not in data.columns:
        return pd.DataFrame()

    data = data.copy()

    data["microphone"] = (
        data["filename"]
        .apply(extract_microphone)
    )

    rows = []

    for (
        speaker_id,
        microphone
    ), group in data.groupby(
        [
            "speaker_id",
            "microphone",
        ]
    ):

        row = {
            "speaker_id":
                speaker_id,

            "speech_status":
                group[
                    "speech_status"
                ].iloc[0],

            "microphone":
                microphone,

            "n_recordings":
                len(group),

            "mean_p_dys_hubert":
                group[
                    "p_dys_hubert"
                ].mean(),

            "mean_p_dys_residual":
                group[
                    "p_dys_residual_hubert"
                ].mean(),

            "mean_delta_p_dys":
                group[
                    "delta_p_dys"
                ].mean(),
        }

        for feature in [
            "rms_mean",
            "rms_std",
            "pause_ratio",
            "f0_mean",
        ]:

            if feature in group.columns:
                row[
                    f"mean_{feature}"
                ] = group[
                    feature
                ].mean()

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# PRINT FOCUS SPEAKERS
# ============================================================

def print_focus_speakers(
    speaker_results,
):

    print("\n" + "=" * 80)
    print(
        "RECORDING CONDITIONS — "
        "FOCUS SPEAKERS"
    )
    print("=" * 80)

    focus = speaker_results[
        speaker_results[
            "speaker_id"
        ].isin(
            FOCUS_SPEAKERS
        )
    ].copy()

    columns = [
        "speaker_id",
        "speech_status",
        "n_recordings",
        "mean_p_dys_hubert",
        "mean_p_dys_residual",
        "mean_delta_p_dys",
        "audio_duration_mean",
        "pause_ratio_mean",
        "f0_mean_mean",
        "rms_mean_mean",
        "rms_std_mean",
    ]

    columns = [
        c
        for c in columns
        if c in focus.columns
    ]

    print(
        focus[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


def print_focus_correlations(
    within_results,
):

    print("\n" + "=" * 80)
    print(
        "WITHIN-SPEAKER CORRELATIONS"
    )
    print("=" * 80)

    focus = within_results[
        within_results[
            "speaker_id"
        ].isin(
            FOCUS_SPEAKERS
        )
    ].copy()

    # Most interesting features first
    focus[
        "abs_corr_delta"
    ] = (
        focus[
            "corr_feature_delta_p_dys"
        ].abs()
    )

    focus = focus.sort_values(
        [
            "speaker_id",
            "abs_corr_delta",
        ],
        ascending=[
            True,
            False,
        ],
    )

    columns = [
        "speaker_id",
        "feature",
        "corr_feature_hubert_p_dys",
        "corr_feature_delta_p_dys",
        "low_change_mean",
        "high_change_mean",
        "high_minus_low",
    ]

    print(
        focus[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading prediction-flip data..."
    )

    data = pd.read_csv(
        FLIPS_PATH
    )

    print(
        f"Recordings: {len(data)}"
    )

    print(
        f"Speakers: "
        f"{data['speaker_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required = [
        "speaker_id",
        "speech_status",
        "fold",
        "true_label",
        "pred_hubert",
        "p_dys_hubert",
        "pred_residual_hubert",
        "p_dys_residual_hubert",
        "delta_p_dys",
        "abs_delta_p_dys",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    speaker_results = (
        analyze_by_speaker(
            data
        )
    )

    within_results = (
        analyze_within_speaker(
            data
        )
    )

    microphone_results = (
        analyze_microphone(
            data
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    speaker_results.to_csv(
        SPEAKER_OUTPUT,
        index=False,
    )

    within_results.to_csv(
        WITHIN_SPEAKER_OUTPUT,
        index=False,
    )

    microphone_results.to_csv(
        MIC_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_focus_speakers(
        speaker_results
    )

    print_focus_correlations(
        within_results
    )

    print("\n" + "=" * 80)
    print("MICROPHONE COUNTS")
    print("=" * 80)

    if len(microphone_results) > 0:

        print(
            microphone_results[
                [
                    "speaker_id",
                    "microphone",
                    "n_recordings",
                    "mean_p_dys_hubert",
                    "mean_delta_p_dys",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    print("\nSaved:")
    print(SPEAKER_OUTPUT)
    print(WITHIN_SPEAKER_OUTPUT)
    print(MIC_OUTPUT)


if __name__ == "__main__":
    main()