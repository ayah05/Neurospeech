from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
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

ACOUSTIC_PATH = (
    ROOT / "data" / "acoustic_features.csv"
)

RESULTS_DIR = ROOT / "results"

FOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "hubert_acoustic_probe_fold_results.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "hubert_acoustic_probe_summary.csv"
)

TEXT_SUMMARY_PATH = (
    RESULTS_DIR
    / "hubert_acoustic_probe_summary.txt"
)


# ============================================================
# CONFIG
# ============================================================

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

RIDGE_ALPHA = 1.0


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

    embeddings = (
        hubert["embeddings"]
        .astype(np.float32)
    )

    dataset_indices = (
        hubert["dataset_indices"]
        .astype(np.int64)
    )

    print(
        f"Embeddings: {embeddings.shape}"
    )

    print(
        f"Indices: {dataset_indices.shape}"
    )

    # --------------------------------------------------------
    # Validate embeddings
    # --------------------------------------------------------

    if embeddings.shape[1] != 768:
        raise ValueError(
            f"Expected 768 dimensions, "
            f"got {embeddings.shape[1]}."
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
            "Duplicate dataset indices "
            "in HuBERT file."
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "HuBERT embeddings contain "
            "NaN or Inf."
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("\nLoading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    required_metadata = {
        "dataset_index",
        "speaker_id",
        "fold",
        "speech_status",
    }

    missing = (
        required_metadata
        - set(metadata.columns)
    )

    if missing:
        raise ValueError(
            f"Missing metadata columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Acoustic features
    # --------------------------------------------------------

    print("Loading acoustic features...")

    acoustic = pd.read_csv(
        ACOUSTIC_PATH
    )

    required_acoustic = {
        "dataset_index",
        *ACOUSTIC_FEATURES,
    }

    missing = (
        required_acoustic
        - set(acoustic.columns)
    )

    if missing:
        raise ValueError(
            f"Missing acoustic columns: "
            f"{sorted(missing)}"
        )

    if acoustic[
        "dataset_index"
    ].duplicated().any():
        raise ValueError(
            "Duplicate dataset_index "
            "in acoustic features."
        )

    # --------------------------------------------------------
    # Merge metadata + acoustic features
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

    print(
        f"Merged samples: {len(data)}"
    )

    if len(data) != len(metadata):
        raise ValueError(
            "Metadata/acoustic merge "
            "lost samples."
        )

    # --------------------------------------------------------
    # Align HuBERT embeddings
    # --------------------------------------------------------

    embedding_lookup = {
        int(dataset_index): row
        for row, dataset_index
        in enumerate(dataset_indices)
    }

    missing_embeddings = [
        int(index)
        for index
        in data[
            "dataset_index"
        ].values
        if int(index)
        not in embedding_lookup
    ]

    if missing_embeddings:
        raise ValueError(
            f"{len(missing_embeddings)} "
            "samples have no HuBERT "
            "embedding."
        )

    embedding_rows = np.asarray(
        [
            embedding_lookup[
                int(index)
            ]
            for index
            in data[
                "dataset_index"
            ].values
        ],
        dtype=np.int64,
    )

    X = embeddings[
        embedding_rows
    ]

    print(
        f"Aligned X: {X.shape}"
    )

    print(
        f"Speakers: "
        f"{data['speaker_id'].nunique()}"
    )

    print(
        f"Folds: "
        f"{sorted(data['fold'].unique())}"
    )

    return data, X


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
                "regressor",
                Ridge(
                    alpha=RIDGE_ALPHA,
                    solver="lsqr",
                ),
            ),
        ]
    )


# ============================================================
# SAFE PEARSON
# ============================================================

def calculate_pearson(
    y_true,
    y_pred,
):

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return np.nan

    correlation, _ = pearsonr(
        y_true,
        y_pred,
    )

    return correlation


# ============================================================
# EVALUATE ONE FEATURE
# ============================================================

def evaluate_feature(
    feature_name,
    data,
    X,
):

    print("\n" + "=" * 70)
    print(
        f"Acoustic target: {feature_name}"
    )
    print("=" * 70)

    results = []

    folds = sorted(
        data["fold"].unique()
    )

    for fold in folds:

        test_mask = (
            data["fold"]
            .to_numpy()
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
                f"Speaker leakage in "
                f"fold {fold}: "
                f"{sorted(overlap)}"
            )

        X_train = X[
            train_mask
        ]

        X_test = X[
            test_mask
        ]

        y_train = (
            data.loc[
                train_mask,
                feature_name,
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_test = (
            data.loc[
                test_mask,
                feature_name,
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        # ----------------------------------------------------
        # Validate target
        # ----------------------------------------------------

        if not np.isfinite(
            y_train
        ).all():
            raise ValueError(
                f"{feature_name}: "
                "non-finite training values."
            )

        if not np.isfinite(
            y_test
        ).all():
            raise ValueError(
                f"{feature_name}: "
                "non-finite test values."
            )

        # ----------------------------------------------------
        # Fit probe
        # ----------------------------------------------------

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        r2 = r2_score(
            y_test,
            y_pred,
        )

        mae = mean_absolute_error(
            y_test,
            y_pred,
        )

        pearson_r = (
            calculate_pearson(
                y_test,
                y_pred,
            )
        )

        results.append(
            {
                "feature":
                    feature_name,
                "fold":
                    int(fold),
                "train_samples":
                    len(X_train),
                "test_samples":
                    len(X_test),
                "r2":
                    r2,
                "mae":
                    mae,
                "pearson_r":
                    pearson_r,
            }
        )

        print(
            f"Fold {fold}: "
            f"R²={r2:.3f} | "
            f"MAE={mae:.4f} | "
            f"r={pearson_r:.3f}"
        )

    return results


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    fold_results,
):

    summary = (
        fold_results
        .groupby(
            "feature",
            as_index=False,
        )
        .agg(
            r2_mean=(
                "r2",
                "mean",
            ),
            r2_std=(
                "r2",
                "std",
            ),
            mae_mean=(
                "mae",
                "mean",
            ),
            mae_std=(
                "mae",
                "std",
            ),
            pearson_r_mean=(
                "pearson_r",
                "mean",
            ),
            pearson_r_std=(
                "pearson_r",
                "std",
            ),
        )
    )

    # Sort by speaker-independent
    # predictive performance.
    summary = (
        summary
        .sort_values(
            "r2_mean",
            ascending=False,
        )
        .reset_index(drop=True)
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
        "HuBERT -> Acoustic Feature "
        "Linear Probing"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "Representation: "
        "facebook/hubert-base-ls960"
    )

    lines.append(
        "Input: 768-D mean-pooled "
        "HuBERT embeddings"
    )

    lines.append(
        "Probe: StandardScaler + "
        f"Ridge(alpha={RIDGE_ALPHA})"
    )

    lines.append(
        "Evaluation: 5-fold "
        "speaker-independent CV"
    )

    lines.append("")

    for _, row in (
        summary.iterrows()
    ):

        lines.append(
            f"{row['feature']}"
        )

        lines.append(
            f"  R²: "
            f"{row['r2_mean']:.3f} "
            f"± {row['r2_std']:.3f}"
        )

        lines.append(
            f"  Pearson r: "
            f"{row['pearson_r_mean']:.3f} "
            f"± "
            f"{row['pearson_r_std']:.3f}"
        )

        lines.append(
            f"  MAE: "
            f"{row['mae_mean']:.4f} "
            f"± "
            f"{row['mae_std']:.4f}"
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

    data, X = load_data()

    all_results = []

    for feature_name in (
        ACOUSTIC_FEATURES
    ):

        feature_results = (
            evaluate_feature(
                feature_name=feature_name,
                data=data,
                X=X,
            )
        )

        all_results.extend(
            feature_results
        )

    fold_results = pd.DataFrame(
        all_results
    )

    summary = build_summary(
        fold_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fold_results.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    text_summary = (
        build_text_summary(
            summary
        )
    )

    TEXT_SUMMARY_PATH.write_text(
        text_summary,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            index=False
        )
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

    print(
        TEXT_SUMMARY_PATH
    )


if __name__ == "__main__":
    main()