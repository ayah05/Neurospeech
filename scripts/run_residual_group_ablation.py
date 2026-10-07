from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    ROOT / "data" / "metadata_with_folds.csv"
)

ACOUSTIC_PATH = (
    ROOT / "data" / "acoustic_features.csv"
)

HUBERT_PATH = (
    ROOT / "data" / "hubert_embeddings.npz"
)

RESULTS_DIR = ROOT / "results"

FOLD_OUTPUT = (
    RESULTS_DIR
    / "residual_group_ablation_fold_results.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "residual_group_ablation_summary.csv"
)

TEXT_OUTPUT = (
    RESULTS_DIR
    / "residual_group_ablation_summary.txt"
)


# ============================================================
# FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {

    "duration_only": [
        "audio_duration",
    ],

    "timing": [
        "voiced_duration",
        "pause_ratio",
        "number_of_pauses",
        "mean_pause_duration",
    ],

    "prosody": [
        "f0_mean",
        "f0_std",
    ],

    "energy": [
        "rms_mean",
        "rms_std",
    ],

    "all_acoustic": [
        "audio_duration",
        "voiced_duration",
        "pause_ratio",
        "number_of_pauses",
        "mean_pause_duration",
        "f0_mean",
        "f0_std",
        "rms_mean",
        "rms_std",
    ],
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading data...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    acoustic = pd.read_csv(
        ACOUSTIC_PATH
    )

    hubert = np.load(
        HUBERT_PATH
    )

    embeddings = hubert[
        "embeddings"
    ]

    dataset_indices = hubert[
        "dataset_indices"
    ]

    print(
        "HuBERT embeddings:",
        embeddings.shape,
    )

    # --------------------------------------------------------
    # Explicit HuBERT mapping
    # --------------------------------------------------------

    embedding_lookup = {
        int(index): embedding
        for index, embedding
        in zip(
            dataset_indices,
            embeddings,
        )
    }

    metadata = metadata.copy()

    metadata["dataset_index"] = (
        metadata["dataset_index"]
        .astype(int)
    )

    acoustic = acoustic.copy()

    acoustic["dataset_index"] = (
        acoustic["dataset_index"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Merge acoustic features
    # --------------------------------------------------------

    data = metadata.merge(
        acoustic,
        on="dataset_index",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_acoustic"),
    )

    print(
        "Merged recordings:",
        len(data),
    )

    # --------------------------------------------------------
    # Align HuBERT explicitly by dataset_index
    # --------------------------------------------------------

    missing_embeddings = [
        idx
        for idx in data["dataset_index"]
        if idx not in embedding_lookup
    ]

    if missing_embeddings:
        raise ValueError(
            "Missing HuBERT embeddings for "
            f"{len(missing_embeddings)} recordings."
        )

    X_hubert = np.stack([
        embedding_lookup[idx]
        for idx in data["dataset_index"]
    ])

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y = (
        data["speech_status"]
        .map({
            "healthy": 0,
            "dysarthria": 1,
        })
        .to_numpy()
    )

    if np.isnan(y).any():
        raise ValueError(
            "Unknown speech_status detected."
        )

    y = y.astype(int)

    print(
        "Speakers:",
        data["speaker_id"].nunique(),
    )

    print(
        "Healthy:",
        int((y == 0).sum()),
    )

    print(
        "Dysarthria:",
        int((y == 1).sum()),
    )

    return (
        data,
        X_hubert,
        y,
    )


# ============================================================
# CLASSIFIER
# ============================================================

def train_classifier(
    X_train,
    y_train,
    X_test,
):

    classifier = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])

    classifier.fit(
        X_train,
        y_train,
    )

    y_pred = classifier.predict(
        X_test
    )

    y_prob = classifier.predict_proba(
        X_test
    )[:, 1]

    return (
        y_pred,
        y_prob,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
    )

    specificity = recall_score(
        y_true,
        y_pred,
        pos_label=0,
    )

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
            ),

        "auroc":
            roc_auc_score(
                y_true,
                y_prob,
            ),

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,
    }


# ============================================================
# RESIDUALIZE HUBERT
# ============================================================

def residualize_hubert(
    X_acoustic_train,
    X_acoustic_test,
    X_hubert_train,
    X_hubert_test,
):

    # --------------------------------------------------------
    # Fit scalers ONLY on outer-training speakers
    # --------------------------------------------------------

    acoustic_scaler = StandardScaler()

    A_train = acoustic_scaler.fit_transform(
        X_acoustic_train
    )

    A_test = acoustic_scaler.transform(
        X_acoustic_test
    )

    hubert_scaler = StandardScaler()

    H_train = hubert_scaler.fit_transform(
        X_hubert_train
    )

    H_test = hubert_scaler.transform(
        X_hubert_test
    )

    # --------------------------------------------------------
    # Acoustic -> HuBERT linear mapping
    # --------------------------------------------------------

    regressor = Ridge(
        alpha=1.0,
        solver="lsqr",
    )

    regressor.fit(
        A_train,
        H_train,
    )

    H_train_predicted = (
        regressor.predict(
            A_train
        )
    )

    H_test_predicted = (
        regressor.predict(
            A_test
        )
    )

    # --------------------------------------------------------
    # Residual representation
    # --------------------------------------------------------

    H_train_residual = (
        H_train
        - H_train_predicted
    )

    H_test_residual = (
        H_test
        - H_test_predicted
    )

    return (
        H_train_residual,
        H_test_residual,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(
    data,
    X_hubert,
    y,
):

    fold_rows = []

    folds = sorted(
        data["fold"].unique()
    )

    for fold in folds:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FOLD {fold}"
        )

        print(
            "=" * 70
        )

        train_mask = (
            data["fold"].to_numpy()
            != fold
        )

        test_mask = (
            data["fold"].to_numpy()
            == fold
        )

        # ----------------------------------------------------
        # Speaker leakage check
        # ----------------------------------------------------

        train_speakers = set(
            data.loc[
                train_mask,
                "speaker_id",
            ]
        )

        test_speakers = set(
            data.loc[
                test_mask,
                "speaker_id",
            ]
        )

        overlap = (
            train_speakers
            & test_speakers
        )

        if overlap:
            raise ValueError(
                "Speaker leakage detected: "
                f"{overlap}"
            )

        print(
            "Train speakers:",
            len(train_speakers),
        )

        print(
            "Test speakers:",
            len(test_speakers),
        )

        y_train = y[
            train_mask
        ]

        y_test = y[
            test_mask
        ]

        H_train = X_hubert[
            train_mask
        ]

        H_test = X_hubert[
            test_mask
        ]

        # ----------------------------------------------------
        # Baseline HuBERT once per fold
        # ----------------------------------------------------

        baseline_pred, baseline_prob = (
            train_classifier(
                H_train,
                y_train,
                H_test,
            )
        )

        baseline_metrics = (
            calculate_metrics(
                y_test,
                baseline_pred,
                baseline_prob,
            )
        )

        baseline_row = {
            "fold": fold,
            "condition": "hubert_original",
            "n_features_removed": 0,
            **baseline_metrics,
        }

        fold_rows.append(
            baseline_row
        )

        print(
            "\nHuBERT original:"
            f" BA={baseline_metrics['balanced_accuracy']:.3f}"
            f" AUROC={baseline_metrics['auroc']:.3f}"
        )

        # ----------------------------------------------------
        # Each acoustic feature group
        # ----------------------------------------------------

        for (
            group_name,
            features
        ) in FEATURE_GROUPS.items():

            print(
                f"\nResidualizing: "
                f"{group_name}"
            )

            X_acoustic = (
                data[
                    features
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            if not np.isfinite(
                X_acoustic
            ).all():

                raise ValueError(
                    "Non-finite values "
                    f"in {group_name}"
                )

            A_train = X_acoustic[
                train_mask
            ]

            A_test = X_acoustic[
                test_mask
            ]

            (
                H_train_residual,
                H_test_residual,
            ) = residualize_hubert(
                A_train,
                A_test,
                H_train,
                H_test,
            )

            y_pred, y_prob = (
                train_classifier(
                    H_train_residual,
                    y_train,
                    H_test_residual,
                )
            )

            metrics = (
                calculate_metrics(
                    y_test,
                    y_pred,
                    y_prob,
                )
            )

            row = {
                "fold": fold,
                "condition":
                    f"residual_{group_name}",
                "n_features_removed":
                    len(features),
                **metrics,
            }

            fold_rows.append(
                row
            )

            print(
                f"  BA="
                f"{metrics['balanced_accuracy']:.3f}"
                f" | AUROC="
                f"{metrics['auroc']:.3f}"
                f" | F1="
                f"{metrics['macro_f1']:.3f}"
            )

    return pd.DataFrame(
        fold_rows
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(
    fold_results,
):

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "sensitivity",
        "specificity",
    ]

    rows = []

    for condition, group in (
        fold_results.groupby(
            "condition",
            sort=False,
        )
    ):

        row = {
            "condition":
                condition,

            "n_features_removed":
                int(
                    group[
                        "n_features_removed"
                    ].iloc[0]
                ),
        }

        for metric in metrics:

            row[
                f"{metric}_mean"
            ] = group[
                metric
            ].mean()

            row[
                f"{metric}_std"
            ] = group[
                metric
            ].std()

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Compare every condition with original HuBERT
    # --------------------------------------------------------

    baseline = summary[
        summary["condition"]
        == "hubert_original"
    ].iloc[0]

    summary[
        "delta_balanced_accuracy"
    ] = (
        summary[
            "balanced_accuracy_mean"
        ]
        - baseline[
            "balanced_accuracy_mean"
        ]
    )

    summary[
        "delta_auroc"
    ] = (
        summary[
            "auroc_mean"
        ]
        - baseline[
            "auroc_mean"
        ]
    )

    summary[
        "delta_macro_f1"
    ] = (
        summary[
            "macro_f1_mean"
        ]
        - baseline[
            "macro_f1_mean"
        ]
    )

    return summary


# ============================================================
# TEXT SUMMARY
# ============================================================

def save_text_summary(
    summary,
):

    lines = []

    lines.append(
        "Residual Group Ablation"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    for _, row in summary.iterrows():

        lines.append(
            row["condition"]
        )

        lines.append(
            f"  Balanced accuracy: "
            f"{row['balanced_accuracy_mean']:.3f}"
            f" ± "
            f"{row['balanced_accuracy_std']:.3f}"
        )

        lines.append(
            f"  Macro F1: "
            f"{row['macro_f1_mean']:.3f}"
            f" ± "
            f"{row['macro_f1_std']:.3f}"
        )

        lines.append(
            f"  AUROC: "
            f"{row['auroc_mean']:.3f}"
            f" ± "
            f"{row['auroc_std']:.3f}"
        )

        lines.append(
            f"  Δ BA vs HuBERT: "
            f"{row['delta_balanced_accuracy']:+.3f}"
        )

        lines.append(
            f"  Δ AUROC vs HuBERT: "
            f"{row['delta_auroc']:+.3f}"
        )

        lines.append("")

    TEXT_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        data,
        X_hubert,
        y,
    ) = load_data()

    fold_results = (
        run_experiment(
            data,
            X_hubert,
            y,
        )
    )

    summary = (
        summarize_results(
            fold_results
        )
    )

    fold_results.to_csv(
        FOLD_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    save_text_summary(
        summary
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 70
    )

    columns = [
        "condition",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "auroc_mean",
        "auroc_std",
        "delta_balanced_accuracy",
        "delta_auroc",
    ]

    print(
        summary[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    print("\nSaved:")
    print(FOLD_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(TEXT_OUTPUT)


if __name__ == "__main__":
    main()