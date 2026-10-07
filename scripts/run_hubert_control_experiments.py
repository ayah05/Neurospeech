from pathlib import Path

import numpy as np
import pandas as pd

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

EMBEDDINGS_PATH = (
    ROOT / "data" / "hubert_embeddings.npz"
)

METADATA_PATH = (
    ROOT / "data" / "metadata_with_folds.csv"
)

WHISPER_PATH = (
    ROOT / "data" / "whisper_full_predictions.csv"
)

STRICT_SUBSET_PATH = (
    ROOT / "data" / "strict_shared_prompt_subset.csv"
)

RESULTS_DIR = ROOT / "results"

PERFECT_FOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "hubert_perfect_asr_fold_results.csv"
)

STRICT_FOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "hubert_strict_shared_prompt_fold_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "hubert_control_experiments_summary.txt"
)

COMPARISON_PATH = (
    RESULTS_DIR
    / "hubert_control_experiments_comparison.csv"
)


# ============================================================
# CONFIG
# ============================================================

LABEL_MAP = {
    "healthy": 0,
    "dysarthria": 1,
}

RANDOM_STATE = 42

EXPECTED_EMBEDDING_DIM = 768


# ============================================================
# LOAD HUBERT EMBEDDINGS
# ============================================================

def load_hubert_embeddings():

    print("=" * 70)
    print("Loading HuBERT embeddings")
    print("=" * 70)

    data = np.load(
        EMBEDDINGS_PATH
    )

    embeddings = (
        data["embeddings"]
        .astype(np.float32)
    )

    dataset_indices = (
        data["dataset_indices"]
        .astype(np.int64)
    )

    print(
        f"Embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"Dataset indices: "
        f"{dataset_indices.shape}"
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "HuBERT embeddings must be 2D."
        )

    if (
        embeddings.shape[1]
        != EXPECTED_EMBEDDING_DIM
    ):
        raise ValueError(
            "Unexpected embedding "
            f"dimension: "
            f"{embeddings.shape[1]}"
        )

    if (
        len(embeddings)
        != len(dataset_indices)
    ):
        raise ValueError(
            "Embedding/index length mismatch."
        )

    if len(
        np.unique(dataset_indices)
    ) != len(dataset_indices):
        raise ValueError(
            "Duplicate HuBERT dataset indices."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "HuBERT embeddings contain "
            "NaN or Inf."
        )

    embedding_lookup = {
        int(dataset_index): row
        for row, dataset_index
        in enumerate(dataset_indices)
    }

    return embeddings, embedding_lookup


# ============================================================
# LOAD BASE METADATA
# ============================================================

def load_metadata():

    metadata = pd.read_csv(
        METADATA_PATH
    )

    required_columns = {
        "dataset_index",
        "speaker_id",
        "speech_status",
        "fold",
    }

    missing = (
        required_columns
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            "Metadata missing columns: "
            f"{sorted(missing)}"
        )

    if metadata[
        "dataset_index"
    ].duplicated().any():
        raise ValueError(
            "Duplicate dataset_index "
            "in metadata."
        )

    return metadata


# ============================================================
# PERFECT-ASR SUBSET
# ============================================================

def build_perfect_asr_subset(
    metadata,
):

    print("\n" + "=" * 70)
    print("Building Perfect-ASR subset")
    print("=" * 70)

    whisper = pd.read_csv(
        WHISPER_PATH
    )

    required_columns = {
        "dataset_index",
        "wer",
        "error",
    }

    missing = (
        required_columns
        - set(whisper.columns)
    )

    if missing:
        raise ValueError(
            "Whisper predictions missing "
            f"columns: {sorted(missing)}"
        )

    # ---------------------------------------------
    # Successful Whisper predictions only
    # ---------------------------------------------

    if whisper[
        "error"
    ].dtype == object:

        successful = (
            whisper["error"].isna()
            | (
                whisper["error"]
                .astype(str)
                .str.strip()
                .isin(["", "nan", "None"])
            )
        )

    else:

        successful = (
            whisper["error"].isna()
        )

    perfect_indices = set(
        whisper.loc[
            successful
            & whisper["wer"].notna()
            & np.isclose(
                whisper["wer"],
                0.0,
                atol=1e-12,
            ),
            "dataset_index",
        ]
        .astype(int)
        .tolist()
    )

    subset = (
        metadata[
            metadata["dataset_index"]
            .isin(perfect_indices)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"Perfect-ASR samples: "
        f"{len(subset)}"
    )

    print(
        "Class counts:"
    )

    print(
        subset[
            "speech_status"
        ].value_counts()
    )

    print(
        f"Speakers: "
        f"{subset['speaker_id'].nunique()}"
    )

    return subset


# ============================================================
# STRICT SHARED-PROMPT SUBSET
# ============================================================

def build_strict_subset(
    metadata,
):

    print("\n" + "=" * 70)
    print(
        "Loading Strict Shared-Prompt "
        "Perfect-ASR subset"
    )
    print("=" * 70)

    strict = pd.read_csv(
        STRICT_SUBSET_PATH
    )

    if (
        "dataset_index"
        not in strict.columns
    ):
        raise ValueError(
            "Strict subset has no "
            "dataset_index column."
        )

    if strict[
        "dataset_index"
    ].duplicated().any():
        raise ValueError(
            "Duplicate dataset_index "
            "in strict subset."
        )

    strict_indices = set(
        strict[
            "dataset_index"
        ]
        .astype(int)
        .tolist()
    )

    subset = (
        metadata[
            metadata["dataset_index"]
            .isin(strict_indices)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(subset) != len(strict):
        raise ValueError(
            "Strict subset size changed "
            "after metadata join. "
            f"Strict file={len(strict)}, "
            f"metadata match={len(subset)}"
        )

    print(
        f"Strict samples: "
        f"{len(subset)}"
    )

    print(
        "Class counts:"
    )

    print(
        subset[
            "speech_status"
        ].value_counts()
    )

    print(
        f"Speakers: "
        f"{subset['speaker_id'].nunique()}"
    )

    # If available, report number of prompts.
    for prompt_column in [
        "reference_normalized",
        "transcription_normalized",
    ]:

        if prompt_column in strict.columns:

            print(
                f"Unique prompts: "
                f"{strict[prompt_column].nunique()}"
            )

            break

    return subset


# ============================================================
# BUILD X / Y
# ============================================================

def build_xy(
    subset,
    embeddings,
    embedding_lookup,
):

    missing_indices = [
        int(index)
        for index
        in subset[
            "dataset_index"
        ].values
        if int(index)
        not in embedding_lookup
    ]

    if missing_indices:
        raise ValueError(
            f"{len(missing_indices)} "
            "samples have no HuBERT "
            "embedding."
        )

    rows = np.asarray(
        [
            embedding_lookup[
                int(index)
            ]
            for index
            in subset[
                "dataset_index"
            ].values
        ],
        dtype=np.int64,
    )

    X = embeddings[
        rows
    ]

    y_series = (
        subset[
            "speech_status"
        ]
        .map(LABEL_MAP)
    )

    if y_series.isna().any():

        unknown = (
            subset.loc[
                y_series.isna(),
                "speech_status",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown labels: {unknown}"
        )

    y = y_series.to_numpy(
        dtype=np.int64
    )

    return X, y


# ============================================================
# MODEL
# ============================================================

def create_model():

    # IMPORTANT:
    # Keep this identical to the Full-TORGO
    # HuBERT baseline.
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
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

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
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),
        "auroc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ============================================================
# CV
# ============================================================

def evaluate_condition(
    name,
    subset,
    embeddings,
    embedding_lookup,
):

    print("\n" + "#" * 70)
    print(name)
    print("#" * 70)

    X, y = build_xy(
        subset=subset,
        embeddings=embeddings,
        embedding_lookup=embedding_lookup,
    )

    print(
        f"X shape: {X.shape}"
    )

    print(
        f"Healthy: "
        f"{np.sum(y == 0)}"
    )

    print(
        f"Dysarthria: "
        f"{np.sum(y == 1)}"
    )

    results = []

    folds = sorted(
        subset["fold"].unique()
    )

    if len(folds) != 5:
        raise ValueError(
            "Expected exactly 5 folds, "
            f"found {folds}."
        )

    for fold in folds:

        print(
            f"\n{'-' * 70}"
        )

        print(
            f"{name} - Fold {fold}"
        )

        print(
            f"{'-' * 70}"
        )

        test_mask = (
            subset[
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
            subset.loc[
                train_mask,
                "speaker_id",
            ].unique()
        )

        test_speakers = set(
            subset.loc[
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
                "Speaker leakage detected "
                f"in fold {fold}: "
                f"{sorted(overlap)}"
            )

        if len(
            np.unique(y_train)
        ) != 2:
            raise ValueError(
                f"Fold {fold}: training "
                "set does not contain "
                "both classes."
            )

        if len(
            np.unique(y_test)
        ) != 2:
            raise ValueError(
                f"Fold {fold}: test set "
                "does not contain "
                "both classes."
            )

        print(
            f"Train: {len(X_train)}"
        )

        print(
            f"Test: {len(X_test)}"
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
            "Train classes: "
            f"healthy={np.sum(y_train == 0)}, "
            f"dysarthria={np.sum(y_train == 1)}"
        )

        print(
            "Test classes: "
            f"healthy={np.sum(y_test == 0)}, "
            f"dysarthria={np.sum(y_test == 1)}"
        )

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_test
        )

        y_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_probability=y_probability,
        )

        results.append(
            {
                "condition": name,
                "fold": int(fold),
                "train_samples":
                    len(X_train),
                "test_samples":
                    len(X_test),
                "train_speakers":
                    len(train_speakers),
                "test_speakers":
                    len(test_speakers),
                **metrics,
            }
        )

        print(
            f"Accuracy: "
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
        results
    )


# ============================================================
# SUMMARIZE CONDITION
# ============================================================

def summarize_condition(
    condition_name,
    results,
    n_samples,
):

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "sensitivity",
        "specificity",
    ]

    summary = {
        "condition": condition_name,
        "samples": n_samples,
    }

    for metric in metrics:

        summary[
            f"{metric}_mean"
        ] = results[
            metric
        ].mean()

        summary[
            f"{metric}_std"
        ] = results[
            metric
        ].std()

    return summary


# ============================================================
# TEXT SUMMARY
# ============================================================

def build_text_summary(
    comparison,
):

    lines = []

    lines.append(
        "HuBERT Control Experiments"
    )

    lines.append(
        "=" * 70
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

    for _, row in (
        comparison.iterrows()
    ):

        lines.append(
            row["condition"]
        )

        lines.append(
            "-" * 50
        )

        lines.append(
            f"Samples: "
            f"{int(row['samples'])}"
        )

        lines.append(
            "Balanced accuracy: "
            f"{row['balanced_accuracy_mean']:.3f} "
            f"± "
            f"{row['balanced_accuracy_std']:.3f}"
        )

        lines.append(
            "Macro F1: "
            f"{row['macro_f1_mean']:.3f} "
            f"± "
            f"{row['macro_f1_std']:.3f}"
        )

        lines.append(
            "AUROC: "
            f"{row['auroc_mean']:.3f} "
            f"± "
            f"{row['auroc_std']:.3f}"
        )

        lines.append(
            "Sensitivity: "
            f"{row['sensitivity_mean']:.3f} "
            f"± "
            f"{row['sensitivity_std']:.3f}"
        )

        lines.append(
            "Specificity: "
            f"{row['specificity_mean']:.3f} "
            f"± "
            f"{row['specificity_std']:.3f}"
        )

        lines.append("")

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

    embeddings, embedding_lookup = (
        load_hubert_embeddings()
    )

    metadata = load_metadata()

    # ========================================================
    # CONDITION 1:
    # PERFECT ASR
    # ========================================================

    perfect_subset = (
        build_perfect_asr_subset(
            metadata
        )
    )

    perfect_results = (
        evaluate_condition(
            name="Perfect ASR",
            subset=perfect_subset,
            embeddings=embeddings,
            embedding_lookup=(
                embedding_lookup
            ),
        )
    )

    perfect_results.to_csv(
        PERFECT_FOLD_RESULTS_PATH,
        index=False,
    )

    # ========================================================
    # CONDITION 2:
    # STRICT SHARED PROMPTS
    # ========================================================

    strict_subset = (
        build_strict_subset(
            metadata
        )
    )

    strict_results = (
        evaluate_condition(
            name=(
                "Strict Shared-Prompt "
                "Perfect ASR"
            ),
            subset=strict_subset,
            embeddings=embeddings,
            embedding_lookup=(
                embedding_lookup
            ),
        )
    )

    strict_results.to_csv(
        STRICT_FOLD_RESULTS_PATH,
        index=False,
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    summaries = [
        summarize_condition(
            condition_name="Perfect ASR",
            results=perfect_results,
            n_samples=len(
                perfect_subset
            ),
        ),
        summarize_condition(
            condition_name=(
                "Strict Shared-Prompt "
                "Perfect ASR"
            ),
            results=strict_results,
            n_samples=len(
                strict_subset
            ),
        ),
    ]

    comparison = pd.DataFrame(
        summaries
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    text_summary = (
        build_text_summary(
            comparison
        )
    )

    SUMMARY_PATH.write_text(
        text_summary,
        encoding="utf-8",
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL HUBERT CONTROL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        text_summary
    )

    print(
        "Saved:"
    )

    print(
        PERFECT_FOLD_RESULTS_PATH
    )

    print(
        STRICT_FOLD_RESULTS_PATH
    )

    print(
        COMPARISON_PATH
    )

    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()