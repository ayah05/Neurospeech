from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = ROOT / "data" / "hubert_embeddings.npz"
METADATA_PATH = ROOT / "data" / "metadata_with_folds.csv"
ACOUSTIC_PATH = ROOT / "data" / "acoustic_features.csv"

RESULTS_DIR = ROOT / "results"

FOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "acoustic_residual_analysis_fold_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "acoustic_residual_analysis_summary.csv"
)

TEXT_SUMMARY_PATH = (
    RESULTS_DIR
    / "acoustic_residual_analysis_summary.txt"
)

OOF_PREDICTIONS_PATH = (
    RESULTS_DIR
    / "acoustic_residual_oof_predictions.csv"
)

# ============================================================
# CONFIG
# ============================================================

# Intentionally exclude speech_ratio because:
# pause_ratio ≈ 1 - speech_ratio
#
# audio_duration is retained as a recording/timing variable,
# but results must not be interpreted as purely clinical
# acoustic information.
ACOUSTIC_FEATURES = [
    "audio_duration",
    "voiced_duration",
    "pause_ratio",
    "number_of_pauses",
    "mean_pause_duration",
    "f0_mean",
    "f0_std",
    "rms_mean",
    "rms_std",
]

RIDGE_ALPHA = 1.0

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("Loading data")
    print("=" * 70)

    # --------------------------------------------------------
    # HuBERT
    # --------------------------------------------------------

    hubert = np.load(EMBEDDINGS_PATH)

    embeddings = (
        hubert["embeddings"]
        .astype(np.float32)
    )

    dataset_indices = (
        hubert["dataset_indices"]
        .astype(np.int64)
    )

    print(f"HuBERT embeddings: {embeddings.shape}")

    if embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected 768 HuBERT dimensions, "
            f"got {embeddings.shape[1]}."
        )

    if len(embeddings) != len(dataset_indices):
        raise ValueError(
            "HuBERT embedding/index length mismatch."
        )

    if len(np.unique(dataset_indices)) != len(dataset_indices):
        raise ValueError(
            "Duplicate dataset indices in HuBERT embeddings."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "HuBERT embeddings contain NaN or Inf."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(METADATA_PATH)

    required_metadata = {
        "dataset_index",
        "speaker_id",
        "speech_status",
        "fold",
    }

    missing = required_metadata - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Acoustic features
    # --------------------------------------------------------

    acoustic = pd.read_csv(ACOUSTIC_PATH)

    required_acoustic = {
        "dataset_index",
        *ACOUSTIC_FEATURES,
    }

    missing = required_acoustic - set(acoustic.columns)

    if missing:
        raise ValueError(
            f"Missing acoustic columns: {sorted(missing)}"
        )

    if acoustic["dataset_index"].duplicated().any():
        raise ValueError(
            "Duplicate dataset_index in acoustic features."
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    data = metadata.merge(
        acoustic[
            [
                "dataset_index",
                *ACOUSTIC_FEATURES,
            ]
        ],
        on="dataset_index",
        how="inner",
        validate="one_to_one",
    )

    if len(data) != len(metadata):
        raise ValueError(
            "Metadata/acoustic merge lost samples."
        )

    # --------------------------------------------------------
    # Align HuBERT
    # --------------------------------------------------------

    embedding_lookup = {
        int(dataset_index): row
        for row, dataset_index
        in enumerate(dataset_indices)
    }

    missing_embeddings = [
        int(index)
        for index in data["dataset_index"].values
        if int(index) not in embedding_lookup
    ]

    if missing_embeddings:
        raise ValueError(
            f"{len(missing_embeddings)} samples "
            "have no HuBERT embedding."
        )

    rows = np.asarray(
        [
            embedding_lookup[int(index)]
            for index in data["dataset_index"].values
        ],
        dtype=np.int64,
    )

    X_hubert = embeddings[rows]

    X_acoustic = (
        data[ACOUSTIC_FEATURES]
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(X_acoustic).all():
        raise ValueError(
            "Acoustic features contain NaN or Inf."
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    label_map = {
        "healthy": 0,
        "dysarthria": 1,
    }

    unknown_labels = (
        set(data["speech_status"].unique())
        - set(label_map)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown speech-status labels: "
            f"{unknown_labels}"
        )

    y = (
        data["speech_status"]
        .map(label_map)
        .to_numpy(dtype=np.int64)
    )

    print(f"Samples: {len(data)}")
    print(f"Speakers: {data['speaker_id'].nunique()}")
    print(f"Healthy: {(y == 0).sum()}")
    print(f"Dysarthria: {(y == 1).sum()}")
    print(f"Acoustic matrix: {X_acoustic.shape}")
    print(f"HuBERT matrix: {X_hubert.shape}")

    return (
        data,
        X_acoustic,
        X_hubert,
        y,
    )


# ============================================================
# CLASSIFIER
# ============================================================

def create_classifier():

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def evaluate_classifier(
    representation_name,
    X_train,
    X_test,
    y_train,
    y_test,
    fold,
):

    model = create_classifier()

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

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
        y_prob,
    )

    dys_mask = y_test == 1
    healthy_mask = y_test == 0

    sensitivity = (
        (y_pred[dys_mask] == 1).mean()
    )

    specificity = (
        (y_pred[healthy_mask] == 0).mean()
    )

    print(
        f"{representation_name:<25} "
        f"BA={balanced_accuracy:.3f} | "
        f"F1={macro_f1:.3f} | "
        f"AUROC={auroc:.3f}"
    )

    metrics = {
        "fold": int(fold),
        "representation": representation_name,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

    # Return predictions as well as aggregate metrics.
    return metrics, y_pred, y_prob


# ============================================================
# RESIDUALIZE HuBERT
# ============================================================

def residualize_hubert(
    X_acoustic_train,
    X_acoustic_test,
    X_hubert_train,
    X_hubert_test,
):

    """
    Estimate the component of HuBERT that is linearly
    predictable from the selected acoustic features.

    IMPORTANT:
    Everything is fitted on TRAIN speakers only.

    The resulting residual representation is:

        H_residual = H - H_predicted_from_acoustics

    This does NOT imply that all acoustic information has
    been removed from HuBERT.
    """

    # --------------------------------------------------------
    # Scale acoustic predictors using TRAIN only
    # --------------------------------------------------------

    acoustic_scaler = StandardScaler()

    A_train = acoustic_scaler.fit_transform(
        X_acoustic_train
    )

    A_test = acoustic_scaler.transform(
        X_acoustic_test
    )

    # --------------------------------------------------------
    # Scale HuBERT target using TRAIN only
    #
    # This is important because different HuBERT dimensions
    # may have different variances.
    # --------------------------------------------------------

    hubert_scaler = StandardScaler()

    H_train = hubert_scaler.fit_transform(
        X_hubert_train
    )

    H_test = hubert_scaler.transform(
        X_hubert_test
    )

    # --------------------------------------------------------
    # Acoustic -> HuBERT multi-output Ridge
    # --------------------------------------------------------

    ridge = Ridge(
        alpha=RIDGE_ALPHA,
        solver="lsqr",
    )

    ridge.fit(
        A_train,
        H_train,
    )

    H_train_pred = ridge.predict(
        A_train
    )

    H_test_pred = ridge.predict(
        A_test
    )

    # --------------------------------------------------------
    # Residual representation
    # --------------------------------------------------------

    H_train_residual = (
        H_train - H_train_pred
    )

    H_test_residual = (
        H_test - H_test_pred
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
    X_acoustic,
    X_hubert,
    y,
):

    results = []
    oof_rows = []

    folds = sorted(
        data["fold"].unique()
    )

    for fold in folds:

        print("\n" + "=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        test_mask = (
            data["fold"].to_numpy()
            == fold
        )

        train_mask = ~test_mask

        # ----------------------------------------------------
        # Speaker leakage check
        # ----------------------------------------------------

        train_speakers = set(
            data.loc[
                train_mask,
                "speaker_id",
            ].unique()
        )

        test_speakers = set(
            data.loc[
                test_mask,
                "speaker_id",
            ].unique()
        )

        overlap = (
            train_speakers
            & test_speakers
        )

        if overlap:
            raise ValueError(
                f"Speaker leakage in fold {fold}: "
                f"{sorted(overlap)}"
            )

        print(
            f"Train speakers: {len(train_speakers)}"
        )

        print(
            f"Test speakers:  {len(test_speakers)}"
        )

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        A_train = X_acoustic[
            train_mask
        ]

        A_test = X_acoustic[
            test_mask
        ]

        H_train = X_hubert[
            train_mask
        ]

        H_test = X_hubert[
            test_mask
        ]

        y_train = y[
            train_mask
        ]

        y_test = y[
            test_mask
        ]

        test_data = (
            data.loc[test_mask]
            .copy()
            .reset_index(drop=True)
        )

        # ====================================================
        # 1. ACOUSTIC
        # ====================================================

        (
            metrics_acoustic,
            pred_acoustic,
            prob_acoustic,
        ) = evaluate_classifier(
            representation_name="Acoustic",
            X_train=A_train,
            X_test=A_test,
            y_train=y_train,
            y_test=y_test,
            fold=fold,
        )

        results.append(
            metrics_acoustic
        )

        # ====================================================
        # 2. HuBERT
        # ====================================================

        (
            metrics_hubert,
            pred_hubert,
            prob_hubert,
        ) = evaluate_classifier(
            representation_name="HuBERT",
            X_train=H_train,
            X_test=H_test,
            y_train=y_train,
            y_test=y_test,
            fold=fold,
        )

        results.append(
            metrics_hubert
        )

        # ====================================================
        # 3. HuBERT + ACOUSTIC
        # ====================================================

        combined_train = np.concatenate(
            [
                H_train,
                A_train,
            ],
            axis=1,
        )

        combined_test = np.concatenate(
            [
                H_test,
                A_test,
            ],
            axis=1,
        )

        (
            metrics_combined,
            pred_combined,
            prob_combined,
        ) = evaluate_classifier(
            representation_name="HuBERT + Acoustic",
            X_train=combined_train,
            X_test=combined_test,
            y_train=y_train,
            y_test=y_test,
            fold=fold,
        )

        results.append(
            metrics_combined
        )

        # ====================================================
        # 4. RESIDUAL HuBERT
        # ====================================================

        (
            H_train_residual,
            H_test_residual,
        ) = residualize_hubert(
            X_acoustic_train=A_train,
            X_acoustic_test=A_test,
            X_hubert_train=H_train,
            X_hubert_test=H_test,
        )

        (
            metrics_residual,
            pred_residual,
            prob_residual,
        ) = evaluate_classifier(
            representation_name="Residual HuBERT",
            X_train=H_train_residual,
            X_test=H_test_residual,
            y_train=y_train,
            y_test=y_test,
            fold=fold,
        )

        results.append(
            metrics_residual
        )

        # ====================================================
        # SAVE PER-AUDIO OUT-OF-FOLD PREDICTIONS
        # ====================================================

        for i in range(len(test_data)):

            row = test_data.iloc[i]

            oof_rows.append(
                {
                    "dataset_index": int(
                        row["dataset_index"]
                    ),
                    "speaker_id": row[
                        "speaker_id"
                    ],
                    "fold": int(fold),
                    "speech_status": row[
                        "speech_status"
                    ],
                    "true_label": int(
                        y_test[i]
                    ),

                    # Acoustic
                    "pred_acoustic": int(
                        pred_acoustic[i]
                    ),
                    "p_dys_acoustic": float(
                        prob_acoustic[i]
                    ),

                    # HuBERT
                    "pred_hubert": int(
                        pred_hubert[i]
                    ),
                    "p_dys_hubert": float(
                        prob_hubert[i]
                    ),

                    # HuBERT + Acoustic
                    "pred_hubert_acoustic": int(
                        pred_combined[i]
                    ),
                    "p_dys_hubert_acoustic": float(
                        prob_combined[i]
                    ),

                    # Residual HuBERT
                    "pred_residual_hubert": int(
                        pred_residual[i]
                    ),
                    "p_dys_residual_hubert": float(
                        prob_residual[i]
                    ),
                }
            )

    fold_results = pd.DataFrame(
        results
    )

    oof_predictions = pd.DataFrame(
        oof_rows
    )

    return (
        fold_results,
        oof_predictions,
    )

# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    fold_results,
):

    summary = (
        fold_results
        .groupby(
            "representation",
            as_index=False,
        )
        .agg(
            balanced_accuracy_mean=(
                "balanced_accuracy",
                "mean",
            ),
            balanced_accuracy_std=(
                "balanced_accuracy",
                "std",
            ),
            macro_f1_mean=(
                "macro_f1",
                "mean",
            ),
            macro_f1_std=(
                "macro_f1",
                "std",
            ),
            auroc_mean=(
                "auroc",
                "mean",
            ),
            auroc_std=(
                "auroc",
                "std",
            ),
            sensitivity_mean=(
                "sensitivity",
                "mean",
            ),
            sensitivity_std=(
                "sensitivity",
                "std",
            ),
            specificity_mean=(
                "specificity",
                "mean",
            ),
            specificity_std=(
                "specificity",
                "std",
            ),
        )
    )

    return summary


# ============================================================
# TEXT SUMMARY
# ============================================================

def build_text_summary(
    summary,
):

    lines = []

    lines.append(
        "Acoustic-Explainable Component Analysis"
    )

    lines.append("=" * 70)

    lines.append(
        "Dataset: Full TORGO"
    )

    lines.append(
        "Evaluation: fixed 5-fold "
        "speaker-independent CV"
    )

    lines.append(
        "Classifier: StandardScaler + "
        "balanced Logistic Regression"
    )

    lines.append(
        "Residualization: "
        "Acoustic -> HuBERT Ridge "
        f"(alpha={RIDGE_ALPHA})"
    )

    lines.append("")

    lines.append(
        "Selected acoustic features:"
    )

    for feature in ACOUSTIC_FEATURES:
        lines.append(
            f"  - {feature}"
        )

    lines.append("")

    for _, row in summary.iterrows():

        lines.append(
            row["representation"]
        )

        lines.append(
            f"  Balanced Accuracy: "
            f"{row['balanced_accuracy_mean']:.3f} "
            f"± "
            f"{row['balanced_accuracy_std']:.3f}"
        )

        lines.append(
            f"  Macro F1: "
            f"{row['macro_f1_mean']:.3f} "
            f"± "
            f"{row['macro_f1_std']:.3f}"
        )

        lines.append(
            f"  AUROC: "
            f"{row['auroc_mean']:.3f} "
            f"± "
            f"{row['auroc_std']:.3f}"
        )

        lines.append(
            f"  Sensitivity: "
            f"{row['sensitivity_mean']:.3f} "
            f"± "
            f"{row['sensitivity_std']:.3f}"
        )

        lines.append(
            f"  Specificity: "
            f"{row['specificity_mean']:.3f} "
            f"± "
            f"{row['specificity_std']:.3f}"
        )

        lines.append("")

    lines.append(
        "Interpretation note:"
    )

    lines.append(
        "Residual HuBERT removes only the "
        "representation component linearly "
        "predictable from the selected acoustic "
        "measurements. It does not guarantee "
        "removal of all acoustic information."
    )

    return "\n".join(lines)


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
        X_acoustic,
        X_hubert,
        y,
    ) = load_data()

    (
        fold_results,
        oof_predictions,
    ) = run_experiment(
        data=data,
        X_acoustic=X_acoustic,
        X_hubert=X_hubert,
        y=y,
    )
    summary = build_summary(
        fold_results
    )

    fold_results.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    text_summary = build_text_summary(
        summary
    )

    TEXT_SUMMARY_PATH.write_text(
        text_summary,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    # ============================================================
    # VALIDATE OOF PREDICTIONS
    # ============================================================

    if len(oof_predictions) != len(data):
        raise ValueError(
            f"Expected {len(data)} OOF predictions, "
            f"got {len(oof_predictions)}."
        )

    if oof_predictions["dataset_index"].duplicated().any():
        raise ValueError(
            "Duplicate dataset_index in OOF predictions."
        )

    expected_indices = set(
        data["dataset_index"].astype(int)
    )

    predicted_indices = set(
        oof_predictions["dataset_index"].astype(int)
    )

    if expected_indices != predicted_indices:
        raise ValueError(
            "OOF predictions do not cover exactly "
            "the full dataset."
        )

    probability_columns = [
        "p_dys_acoustic",
        "p_dys_hubert",
        "p_dys_hubert_acoustic",
        "p_dys_residual_hubert",
    ]

    for column in probability_columns:

        if not oof_predictions[column].between(
                0.0,
                1.0,
        ).all():
            raise ValueError(
                f"Invalid probabilities in {column}."
            )

    oof_predictions.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    print("\nSaved:")
    print(FOLD_RESULTS_PATH)
    print(SUMMARY_PATH)
    print(TEXT_SUMMARY_PATH)
    print(OOF_PREDICTIONS_PATH)


if __name__ == "__main__":
    main()