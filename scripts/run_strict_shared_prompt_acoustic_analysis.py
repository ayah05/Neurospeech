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

SUBSET_PATH = (
    ROOT / "data" / "strict_shared_prompt_subset.csv"
)

ACOUSTIC_PATH = (
    ROOT / "data" / "acoustic_features.csv"
)

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "strict_shared_prompt_acoustic_fold_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "strict_shared_prompt_acoustic_summary.txt"
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

print("Loading strict shared-prompt subset...")

subset = pd.read_csv(
    SUBSET_PATH
)

print(
    f"Strict subset recordings: {len(subset)}"
)


print("\nLoading acoustic features...")

acoustic = pd.read_csv(
    ACOUSTIC_PATH
)

print(
    f"Acoustic recordings: {len(acoustic)}"
)


# ============================================================
# VALIDATION
# ============================================================

if not subset["dataset_index"].is_unique:
    raise ValueError(
        "dataset_index is not unique in strict subset."
    )


if not acoustic["dataset_index"].is_unique:
    raise ValueError(
        "dataset_index is not unique in acoustic features."
    )


missing_features = (
    set(FEATURE_COLUMNS)
    - set(acoustic.columns)
)

if missing_features:
    raise ValueError(
        f"Missing acoustic features: {sorted(missing_features)}"
    )


# ============================================================
# MERGE ACOUSTIC FEATURES
# ============================================================

# Keep metadata from the strict subset and only add
# acoustic feature columns from acoustic_features.csv.

# ============================================================
# MERGE ACOUSTIC FEATURES
# ============================================================
#
# Some columns such as audio_duration may already exist in the
# strict subset because they were carried over from the Whisper
# results.
#
# To avoid pandas creating columns such as:
#
#   audio_duration_x
#   audio_duration_y
#
# we only add acoustic feature columns that are not already
# present in the strict subset.
# ============================================================

features_to_add = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in subset.columns
]


print(
    "\nAcoustic features already present in subset:"
)

already_present = [
    feature
    for feature in FEATURE_COLUMNS
    if feature in subset.columns
]

print(already_present)


print(
    "\nAcoustic features to merge:"
)

print(features_to_add)


acoustic_features = acoustic[
    ["dataset_index"] + features_to_add
].copy()


data = subset.merge(
    acoustic_features,
    on="dataset_index",
    how="inner",
    validate="one_to_one",
)


print(
    f"\nMerged recordings: {len(data)}"
)


if len(data) != len(subset):
    raise RuntimeError(
        "Some strict-subset recordings could not be "
        "matched to acoustic features."
    )


# Final safety check:
# all required features must now exist exactly once.

missing_after_merge = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in data.columns
]


if missing_after_merge:
    raise RuntimeError(
        "Required acoustic features are still missing "
        f"after merge: {missing_after_merge}"
    )

print(
    f"\nMerged recordings: {len(data)}"
)


if len(data) != len(subset):
    raise RuntimeError(
        "Some strict-subset recordings could not be "
        "matched to acoustic features."
    )


# ============================================================
# VERIFY EXPERIMENT CONDITIONS
# ============================================================

if not np.allclose(
    data["wer"].to_numpy(),
    0.0,
):
    raise RuntimeError(
        "Strict subset contains recordings with WER != 0."
    )


print(
    f"Unique prompts: "
    f"{data['reference_normalized'].nunique()}"
)

print(
    f"Unique speakers: "
    f"{data['speaker_id'].nunique()}"
)


# ============================================================
# LABELS
# ============================================================

LABEL_MAP = {
    "healthy": 0,
    "dysarthria": 1,
}


data["label"] = (
    data["speech_status"]
    .map(LABEL_MAP)
)


if data["label"].isna().any():
    raise ValueError(
        "Unexpected speech_status found."
    )


data["label"] = (
    data["label"].astype(int)
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
    "STRICT SHARED-PROMPT ACOUSTIC ANALYSIS"
)

print(
    "========================================"
)


for fold in sorted(
    data["fold"].unique()
):

    print(
        f"\n--- Fold {fold} ---"
    )


    train_data = data[
        data["fold"] != fold
    ].copy()

    test_data = data[
        data["fold"] == fold
    ].copy()


    # --------------------------------------------------------
    # Speaker leakage check
    # --------------------------------------------------------

    train_speakers = set(
        train_data["speaker_id"].unique()
    )

    test_speakers = set(
        test_data["speaker_id"].unique()
    )

    overlap = (
        train_speakers
        & test_speakers
    )


    if overlap:
        raise RuntimeError(
            f"Speaker leakage detected in fold {fold}: "
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
    # Class counts
    # --------------------------------------------------------

    train_counts = (
        train_data["speech_status"]
        .value_counts()
    )

    test_counts = (
        test_data["speech_status"]
        .value_counts()
    )


    print("\nTrain class distribution:")

    print(
        train_counts.to_string()
    )


    print("\nTest class distribution:")

    print(
        test_counts.to_string()
    )


    if train_data["label"].nunique() < 2:
        raise RuntimeError(
            f"Fold {fold}: training set contains "
            "only one class."
        )


    if test_data["label"].nunique() < 2:
        raise RuntimeError(
            f"Fold {fold}: test set contains "
            "only one class."
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
    # TRAIN
    # --------------------------------------------------------

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # PREDICT
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
    # METRICS
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
        f"\nAccuracy:          {accuracy:.3f}"
    )

    print(
        f"Balanced accuracy: {balanced_accuracy:.3f}"
    )

    print(
        f"Macro F1:          {macro_f1:.3f}"
    )

    print(
        f"AUROC:             {auroc:.3f}"
    )

    print(
        f"Sensitivity:       {sensitivity:.3f}"
    )

    print(
        f"Specificity:       {specificity:.3f}"
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
# RESULTS
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
    "Strict Shared-Prompt Acoustic Analysis"
)

summary_lines.append(
    "=" * 45
)

summary_lines.append("")

summary_lines.append(
    f"Recordings: {len(data)}"
)

summary_lines.append(
    f"Prompts: "
    f"{data['reference_normalized'].nunique()}"
)

summary_lines.append(
    f"Speakers: "
    f"{data['speaker_id'].nunique()}"
)

summary_lines.append("")

summary_lines.append(
    "Experimental constraints:"
)

summary_lines.append(
    "  WER = 0 for every recording"
)

summary_lines.append(
    "  >= 3 healthy speakers per prompt"
)

summary_lines.append(
    "  >= 3 dysarthria speakers per prompt"
)

summary_lines.append(
    "  speaker-independent fixed folds"
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
# FINAL OUTPUT
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