from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = ROOT / "data" / "acoustic_features.csv"

RESULTS_DIR = ROOT / "results"

RESULTS_PATH = (
    RESULTS_DIR
    / "acoustic_ablation_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "acoustic_ablation_summary.csv"
)

REPORT_PATH = (
    RESULTS_DIR
    / "acoustic_ablation_report.txt"
)


# ============================================================
# FEATURE GROUPS
# ============================================================

TIMING_FEATURES = [
    "voiced_duration",
    "speech_ratio",
    "pause_ratio",
    "number_of_pauses",
    "mean_pause_duration",
]

PROSODY_FEATURES = [
    "f0_mean",
    "f0_std",
]

ENERGY_FEATURES = [
    "rms_mean",
    "rms_std",
]

ALL_FEATURES = [
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
# ABLATION EXPERIMENTS
# ============================================================

EXPERIMENTS = {
    "duration_only": [
        "audio_duration",
    ],

    "timing": TIMING_FEATURES,

    "prosody": PROSODY_FEATURES,

    "energy": ENERGY_FEATURES,

    "acoustic_without_duration": [
        feature
        for feature in ALL_FEATURES
        if feature != "audio_duration"
    ],

    "all_acoustic": ALL_FEATURES,
}


# ============================================================
# MODEL
# ============================================================

def create_model():
    """
    Create a fresh model pipeline.

    A new model is created for every fold and every
    ablation experiment.
    """

    return Pipeline(
        [
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
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# EVALUATE ONE FOLD
# ============================================================

def evaluate_fold(
    train_df,
    test_df,
    feature_columns,
    experiment_name,
    fold_number,
):

    X_train = train_df[
        feature_columns
    ]

    X_test = test_df[
        feature_columns
    ]

    y_train = (
        train_df["speech_status"]
        .map(
            {
                "healthy": 0,
                "dysarthria": 1,
            }
        )
    )

    y_test = (
        test_df["speech_status"]
        .map(
            {
                "healthy": 0,
                "dysarthria": 1,
            }
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # PREDICTIONS
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

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()
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

    return {
        "experiment": experiment_name,
        "fold": fold_number,
        "n_features": len(feature_columns),
        "features": ",".join(
            feature_columns
        ),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_data(data):

    print("Validating dataset...")

    expected_folds = {
        1, 2, 3, 4, 5
    }

    actual_folds = set(
        data["fold"].unique()
    )

    if actual_folds != expected_folds:
        raise ValueError(
            f"Expected folds "
            f"{expected_folds}, "
            f"found {actual_folds}."
        )

    # --------------------------------------------------------
    # CHECK SPEAKER LEAKAGE
    # --------------------------------------------------------

    folds_per_speaker = (
        data
        .groupby("speaker_id")["fold"]
        .nunique()
    )

    if not (
        folds_per_speaker == 1
    ).all():

        raise ValueError(
            "Speaker leakage detected."
        )

    # --------------------------------------------------------
    # CHECK FEATURES
    # --------------------------------------------------------

    required_features = set(
        ALL_FEATURES
    )

    missing_features = (
        required_features
        - set(data.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing acoustic features: "
            f"{missing_features}"
        )

    print(
        "✓ Five folds found"
    )

    print(
        "✓ Every speaker belongs "
        "to exactly one fold"
    )

    print(
        "✓ All required acoustic "
        "features found"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading acoustic features..."
    )

    data = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"Recordings: {len(data)}"
    )

    print(
        f"Speakers:   "
        f"{data['speaker_id'].nunique()}"
    )

    print()

    validate_data(data)

    all_results = []

    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    for (
        experiment_name,
        feature_columns,
    ) in EXPERIMENTS.items():

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"EXPERIMENT: "
            f"{experiment_name}"
        )

        print(
            "=" * 80
        )

        print(
            "Features:"
        )

        for feature in feature_columns:
            print(
                f"  - {feature}"
            )

        fold_results = []

        # ----------------------------------------------------
        # FIVE OUTER FOLDS
        # ----------------------------------------------------

        for fold_number in range(
            1,
            6
        ):

            train_df = data[
                data["fold"]
                != fold_number
            ].copy()

            test_df = data[
                data["fold"]
                == fold_number
            ].copy()

            # ------------------------------------------------
            # LEAKAGE CHECK
            # ------------------------------------------------

            train_speakers = set(
                train_df[
                    "speaker_id"
                ].unique()
            )

            test_speakers = set(
                test_df[
                    "speaker_id"
                ].unique()
            )

            overlap = (
                train_speakers
                & test_speakers
            )

            if overlap:
                raise ValueError(
                    f"Speaker leakage "
                    f"in fold "
                    f"{fold_number}: "
                    f"{overlap}"
                )

            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            result = evaluate_fold(
                train_df=train_df,
                test_df=test_df,
                feature_columns=feature_columns,
                experiment_name=experiment_name,
                fold_number=fold_number,
            )

            fold_results.append(
                result
            )

            all_results.append(
                result
            )

            print(
                f"\nFold {fold_number}"
            )

            print(
                f"  Balanced accuracy: "
                f"{result['balanced_accuracy']:.3f}"
            )

            print(
                f"  Macro F1:          "
                f"{result['macro_f1']:.3f}"
            )

            print(
                f"  AUROC:             "
                f"{result['roc_auc']:.3f}"
            )

        # ----------------------------------------------------
        # EXPERIMENT SUMMARY
        # ----------------------------------------------------

        experiment_df = pd.DataFrame(
            fold_results
        )

        print(
            "\nMean ± std:"
        )

        for metric in [
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
        ]:

            mean = (
                experiment_df[
                    metric
                ].mean()
            )

            std = (
                experiment_df[
                    metric
                ].std(ddof=1)
            )

            print(
                f"  {metric:20s}: "
                f"{mean:.3f} "
                f"± {std:.3f}"
            )


    # ========================================================
    # CREATE FULL RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )


    # ========================================================
    # CREATE SUMMARY TABLE
    # ========================================================

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "sensitivity",
        "specificity",
    ]

    summary_rows = []

    for (
        experiment_name,
        experiment_df,
    ) in results_df.groupby(
        "experiment",
        sort=False,
    ):

        row = {
            "experiment":
                experiment_name,

            "n_features":
                experiment_df[
                    "n_features"
                ].iloc[0],
        }

        for metric in metric_columns:

            row[
                f"{metric}_mean"
            ] = (
                experiment_df[
                    metric
                ].mean()
            )

            row[
                f"{metric}_std"
            ] = (
                experiment_df[
                    metric
                ].std(ddof=1)
            )

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )


    # ========================================================
    # PRINT FINAL COMPARISON
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL ABLATION COMPARISON"
    )

    print(
        "=" * 100
    )

    comparison_columns = [
        "experiment",
        "n_features",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]

    print(
        summary_df[
            comparison_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.3f}"
            ),
        )
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8",
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8",
    )


    # ========================================================
    # TEXT REPORT
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "NEUROSPEECH "
            "ACOUSTIC FEATURE ABLATION\n"
        )

        file.write(
            "=" * 100
            + "\n\n"
        )

        file.write(
            "Evaluation:\n"
        )

        file.write(
            "5-fold speaker-independent "
            "cross-validation\n"
        )

        file.write(
            "Model: Logistic Regression\n"
        )

        file.write(
            "Class weighting: balanced\n\n"
        )

        # --------------------------------------------
        # FEATURE SETS
        # --------------------------------------------

        file.write(
            "FEATURE SETS\n"
        )

        file.write(
            "-" * 100
            + "\n"
        )

        for (
            experiment_name,
            features,
        ) in EXPERIMENTS.items():

            file.write(
                f"\n{experiment_name}\n"
            )

            for feature in features:
                file.write(
                    f"  - {feature}\n"
                )

        # --------------------------------------------
        # SUMMARY
        # --------------------------------------------

        file.write(
            "\n\nSUMMARY\n"
        )

        file.write(
            "-" * 100
            + "\n"
        )

        file.write(
            summary_df[
                comparison_columns
            ].to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        # --------------------------------------------
        # PER-FOLD RESULTS
        # --------------------------------------------

        file.write(
            "\n\nPER-FOLD RESULTS\n"
        )

        file.write(
            "-" * 100
            + "\n"
        )

        file.write(
            results_df[
                [
                    "experiment",
                    "fold",
                    "balanced_accuracy",
                    "macro_f1",
                    "roc_auc",
                    "sensitivity",
                    "specificity",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.3f}"
                ),
            )
        )

        file.write("\n")


    # ========================================================
    # DONE
    # ========================================================

    print(
        "\nDetailed results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print(
        "\nSummary saved to:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nReport saved to:"
    )

    print(
        REPORT_PATH
    )


if __name__ == "__main__":
    main()