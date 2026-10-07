from pathlib import Path

import numpy as np
import pandas as pd

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

EMBEDDINGS_PATH = (
    ROOT
    / "data"
    / "hubert_embeddings.npz"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "metadata_with_folds.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
)

FOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "hubert_baseline_fold_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "hubert_baseline_summary.txt"
)


# ============================================================
# CONFIG
# ============================================================

LABEL_MAP = {
    "healthy": 0,
    "dysarthria": 1,
}

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("Loading HuBERT embeddings")
    print("=" * 70)

    hubert = np.load(
        EMBEDDINGS_PATH
    )

    embeddings = hubert[
        "embeddings"
    ].astype(
        np.float32
    )

    dataset_indices = hubert[
        "dataset_indices"
    ].astype(
        np.int64
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Dataset indices shape: "
        f"{dataset_indices.shape}"
    )

    # --------------------------------------------------------
    # Validate HuBERT file
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a "
            "2D matrix."
        )

    if embeddings.shape[1] != 768:
        raise ValueError(
            "Expected 768-dimensional "
            f"HuBERT embeddings, got "
            f"{embeddings.shape[1]}."
        )

    if (
        len(embeddings)
        != len(dataset_indices)
    ):
        raise ValueError(
            "Embedding count and "
            "dataset index count differ."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "HuBERT embeddings contain "
            "NaN or Inf."
        )

    if len(
        np.unique(dataset_indices)
    ) != len(dataset_indices):
        raise ValueError(
            "Duplicate dataset indices "
            "found in HuBERT embeddings."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Loading metadata")
    print("=" * 70)

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata rows: "
        f"{len(metadata)}"
    )

    required_columns = {
        "dataset_index",
        "speaker_id",
        "speech_status",
        "fold",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing metadata columns: "
            f"{sorted(missing_columns)}"
        )

    if metadata[
        "dataset_index"
    ].duplicated().any():
        raise ValueError(
            "Duplicate dataset_index "
            "values in metadata."
        )

    # --------------------------------------------------------
    # Map dataset_index -> embedding row
    # --------------------------------------------------------

    embedding_lookup = {
        int(dataset_index): row
        for row, dataset_index
        in enumerate(dataset_indices)
    }

    missing_embeddings = [
        int(index)
        for index
        in metadata[
            "dataset_index"
        ].values
        if int(index)
        not in embedding_lookup
    ]

    if missing_embeddings:
        raise ValueError(
            f"{len(missing_embeddings)} "
            "metadata rows have no "
            "HuBERT embedding."
        )

    # Reorder embeddings to exactly match
    # the metadata dataframe.
    row_indices = np.asarray(
        [
            embedding_lookup[
                int(index)
            ]
            for index
            in metadata[
                "dataset_index"
            ].values
        ],
        dtype=np.int64,
    )

    X = embeddings[
        row_indices
    ]

    y = (
        metadata[
            "speech_status"
        ]
        .map(LABEL_MAP)
    )

    if y.isna().any():
        unknown_labels = (
            metadata.loc[
                y.isna(),
                "speech_status",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unknown speech_status "
            f"labels: {unknown_labels}"
        )

    y = y.to_numpy(
        dtype=np.int64
    )

    print(
        f"\nFinal X shape: "
        f"{X.shape}"
    )

    print(
        f"Final y shape: "
        f"{y.shape}"
    )

    print(
        f"Healthy samples: "
        f"{np.sum(y == 0)}"
    )

    print(
        f"Dysarthria samples: "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Speakers: "
        f"{metadata['speaker_id'].nunique()}"
    )

    print(
        f"Folds: "
        f"{sorted(metadata['fold'].unique())}"
    )

    return metadata, X, y


# ============================================================
# MODEL
# ============================================================

def create_model():

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
                    random_state=(
                        RANDOM_STATE
                    ),
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    auroc = roc_auc_score(
        y_true,
        y_probability,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
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
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "macro_f1": macro_f1,
        "auroc": auroc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    metadata,
    X,
    y,
):

    fold_results = []

    folds = sorted(
        metadata[
            "fold"
        ].unique()
    )

    print("\n" + "=" * 70)
    print(
        "HuBERT speaker-independent "
        "cross-validation"
    )
    print("=" * 70)

    for fold in folds:

        print(
            f"\n{'-' * 70}"
        )

        print(
            f"Fold {fold}"
        )

        print(
            f"{'-' * 70}"
        )

        test_mask = (
            metadata[
                "fold"
            ].to_numpy()
            == fold
        )

        train_mask = ~test_mask

        X_train = X[
            train_mask
        ]

        X_test = X[
            test_mask
        ]

        y_train = y[
            train_mask
        ]

        y_test = y[
            test_mask
        ]

        train_speakers = set(
            metadata.loc[
                train_mask,
                "speaker_id",
            ].unique()
        )

        test_speakers = set(
            metadata.loc[
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
                "Speaker leakage detected: "
                f"{sorted(overlap)}"
            )

        print(
            f"Train samples: "
            f"{len(X_train)}"
        )

        print(
            f"Test samples: "
            f"{len(X_test)}"
        )

        print(
            f"Train speakers: "
            f"{len(train_speakers)}"
        )

        print(
            f"Test speakers: "
            f"{len(test_speakers)}"
        )

        print(
            "Train class counts: "
            f"healthy={np.sum(y_train == 0)}, "
            f"dysarthria={np.sum(y_train == 1)}"
        )

        print(
            "Test class counts: "
            f"healthy={np.sum(y_test == 0)}, "
            f"dysarthria={np.sum(y_test == 1)}"
        )

        # ----------------------------------------------------
        # Train linear probe
        # ----------------------------------------------------

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = model.predict(
            X_test
        )

        y_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = (
            calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_probability=(
                    y_probability
                ),
            )
        )

        result = {
            "fold": fold,
            "train_samples": (
                len(X_train)
            ),
            "test_samples": (
                len(X_test)
            ),
            "train_speakers": (
                len(train_speakers)
            ),
            "test_speakers": (
                len(test_speakers)
            ),
            **metrics,
        }

        fold_results.append(
            result
        )

        print(
            f"\nAccuracy: "
            f"{metrics['accuracy']:.3f}"
        )

        print(
            f"Balanced accuracy: "
            f"{metrics['balanced_accuracy']:.3f}"
        )

        print(
            f"Macro F1: "
            f"{metrics['macro_f1']:.3f}"
        )

        print(
            f"AUROC: "
            f"{metrics['auroc']:.3f}"
        )

        print(
            f"Sensitivity: "
            f"{metrics['sensitivity']:.3f}"
        )

        print(
            f"Specificity: "
            f"{metrics['specificity']:.3f}"
        )

    return pd.DataFrame(
        fold_results
    )


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    results,
    metadata,
):

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "sensitivity",
        "specificity",
    ]

    lines = []

    lines.append(
        "HuBERT Linear Probe - Full TORGO"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"Samples: {len(metadata)}"
    )

    lines.append(
        "Embedding dimension: 768"
    )

    lines.append(
        f"Speakers: "
        f"{metadata['speaker_id'].nunique()}"
    )

    lines.append(
        "Representation: "
        "facebook/hubert-base-ls960"
    )

    lines.append(
        "Classifier: "
        "balanced logistic regression"
    )

    lines.append(
        "Evaluation: "
        "5-fold speaker-independent CV"
    )

    lines.append("")

    for metric in metric_columns:

        mean = results[
            metric
        ].mean()

        std = results[
            metric
        ].std()

        lines.append(
            f"{metric}: "
            f"{mean:.3f} ± "
            f"{std:.3f}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata, X, y = (
        load_data()
    )

    results = (
        run_cross_validation(
            metadata=metadata,
            X=X,
            y=y,
        )
    )

    results.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
    )

    summary = create_summary(
        results=results,
        metadata=metadata,
    )

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MEAN ± STD"
    )

    print(
        "=" * 70
    )

    print(summary)

    print(
        "\nSaved fold results to:"
    )

    print(
        FOLD_RESULTS_PATH
    )

    print(
        "\nSaved summary to:"
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()