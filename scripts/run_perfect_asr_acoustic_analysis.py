from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

ACOUSTIC_PATH = (
    ROOT / "data" / "acoustic_features.csv"
)

WHISPER_PATH = (
    ROOT / "data" / "whisper_full_predictions.csv"
)

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "perfect_asr_acoustic_fold_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "perfect_asr_acoustic_summary.txt"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
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
# LOAD DATA
# ============================================================

print("Loading acoustic features...")

acoustic = pd.read_csv(
    ACOUSTIC_PATH
)

print(
    f"Acoustic recordings: {len(acoustic)}"
)


print("\nLoading Whisper predictions...")

whisper = pd.read_csv(
    WHISPER_PATH
)

print(
    f"Whisper recordings: {len(whisper)}"
)


# ============================================================
# VALIDATE INPUTS
# ============================================================

if not acoustic["dataset_index"].is_unique:
    raise ValueError(
        "dataset_index is not unique "
        "in acoustic_features.csv"
    )


if not whisper["dataset_index"].is_unique:
    raise ValueError(
        "dataset_index is not unique "
        "in whisper_full_predictions.csv"
    )


missing_features = (
    set(FEATURE_COLUMNS)
    - set(acoustic.columns)
)

if missing_features:
    raise ValueError(
        "Missing acoustic features: "
        f"{sorted(missing_features)}"
    )


# ============================================================
# SELECT WHISPER COLUMNS
# ============================================================

whisper_subset = whisper[
    [
        "dataset_index",
        "wer",
        "cer",
        "error",
    ]
].copy()


# ============================================================
# MERGE
# ============================================================

data = acoustic.merge(
    whisper_subset,
    on="dataset_index",
    how="inner",
    validate="one_to_one",
)


print(
    f"\nMerged recordings: {len(data)}"
)


if len(data) != len(acoustic):
    print(
        "WARNING: merged dataset size differs "
        "from acoustic dataset size."
    )


# ============================================================
# KEEP SUCCESSFUL WHISPER RECORDINGS
# ============================================================

data = data[
    data["error"].isna()
].copy()


# ============================================================
# PERFECT-ASR SUBSET
# ============================================================
#
# WER == 0 means Whisper produced a perfect normalized
# word-level transcription according to our evaluation.
#
# np.isclose is used rather than direct floating-point
# equality.
# ============================================================

perfect = data[
    np.isclose(
        data["wer"],
        0.0,
    )
].copy()


print(
    "\n========================================"
)
print(
    "PERFECT-ASR SUBSET"
)
print(
    "========================================"
)

print(
    f"Total recordings: {len(perfect)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = (
    perfect[
        "speech_status"
    ]
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

speaker_counts = (
    perfect
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
    "\nPerfect-ASR recordings per speaker:"
)

print(
    speaker_counts.to_string(
        index=False
    )
)


# ============================================================
# LABELS
# ============================================================
#
# healthy     = 0
# dysarthria  = 1
# ============================================================

LABEL_MAP = {
    "healthy": 0,
    "dysarthria": 1,
}


perfect["label"] = (
    perfect[
        "speech_status"
    ]
    .map(
        LABEL_MAP
    )
)


if perfect["label"].isna().any():
    raise ValueError(
        "Unexpected speech_status found."
    )


perfect["label"] = (
    perfect["label"].astype(int)
)


# ============================================================
# MODEL
# ============================================================

def create_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

fold_results = []


print(
    "\n========================================"
)
print(
    "5-FOLD SPEAKER-INDEPENDENT EVALUATION"
)
print(
    "========================================"
)


for fold in sorted(
    perfect["fold"].unique()
):

    print(
        f"\n--- Fold {fold} ---"
    )


    train_data = perfect[
        perfect["fold"] != fold
    ].copy()

    test_data = perfect[
        perfect["fold"] == fold
    ].copy()


    train_speakers = set(
        train_data[
            "speaker_id"
        ].unique()
    )

    test_speakers = set(
        test_data[
            "speaker_id"
        ].unique()
    )


    overlap = (
        train_speakers
        & test_speakers
    )


    if overlap:
        raise RuntimeError(
            "Speaker leakage detected: "
            f"{sorted(overlap)}"
        )


    print(
        f"Train recordings: {len(train_data)}"
    )

    print(
        f"Test recordings:  {len(test_data)}"
    )

    print(
        f"Train speakers:   {len(train_speakers)}"
    )

    print(
        f"Test speakers:    {len(test_speakers)}"
    )


    # --------------------------------------------------------
    # Check whether both classes exist
    # --------------------------------------------------------

    if (
        train_data["label"].nunique()
        < 2
    ):
        raise RuntimeError(
            f"Fold {fold}: training set "
            "contains only one class."
        )


    if (
        test_data["label"].nunique()
        < 2
    ):
        raise RuntimeError(
            f"Fold {fold}: test set "
            "contains only one class."
        )


    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "label"
    ]


    X_test = test_data[
        FEATURE_COLUMNS
    ]

    y_test = test_data[
        "label"
    ]


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    auroc = roc_auc_score(
        y_test,
        y_probability,
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )
        .ravel()
    )


    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )


    print(
        f"Accuracy:          "
        f"{accuracy:.3f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.3f}"
    )

    print(
        f"Macro F1:          "
        f"{macro_f1:.3f}"
    )

    print(
        f"AUROC:             "
        f"{auroc:.3f}"
    )

    print(
        f"Sensitivity:       "
        f"{sensitivity:.3f}"
    )

    print(
        f"Specificity:       "
        f"{specificity:.3f}"
    )


    fold_results.append(
        {
            "fold": fold,
            "train_recordings":
                len(train_data),

            "test_recordings":
                len(test_data),

            "train_speakers":
                len(train_speakers),

            "test_speakers":
                len(test_speakers),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_f1":
                macro_f1,

            "auroc":
                auroc,

            "sensitivity":
                sensitivity,

            "specificity":
                specificity,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)


fold_results_df.to_csv(
    FOLD_RESULTS_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "sensitivity",
    "specificity",
]


summary_lines = []

summary_lines.append(
    "Perfect-ASR Acoustic Analysis"
)

summary_lines.append(
    "=" * 40
)

summary_lines.append("")

summary_lines.append(
    f"Perfect-ASR recordings: "
    f"{len(perfect)}"
)

summary_lines.append("")


summary_lines.append(
    "Class distribution:"
)

for (
    speech_status,
    count,
) in class_counts.items():

    summary_lines.append(
        f"  {speech_status}: {count}"
    )


summary_lines.append("")

summary_lines.append(
    "Fold results:"
)

summary_lines.append(
    fold_results_df.to_string(
        index=False
    )
)

summary_lines.append("")

summary_lines.append(
    "Mean ± standard deviation:"
)


for metric in METRICS:

    mean_value = (
        fold_results_df[
            metric
        ].mean()
    )

    std_value = (
        fold_results_df[
            metric
        ].std()
    )

    summary_lines.append(
        f"{metric}: "
        f"{mean_value:.3f} "
        f"± {std_value:.3f}"
    )


summary_text = "\n".join(
    summary_lines
)


SUMMARY_PATH.write_text(
    summary_text,
    encoding="utf-8",
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "FINAL SUMMARY"
)

print(
    "========================================"
)

print(
    summary_text
)


print(
    "\nSaved:"
)

print(
    FOLD_RESULTS_PATH
)

print(
    SUMMARY_PATH
)