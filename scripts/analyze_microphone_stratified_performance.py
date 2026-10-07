from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OOF_PATH = (
    ROOT
    / "results"
    / "acoustic_residual_oof_predictions.csv"
)

METADATA_PATH = (
    ROOT
    / "data"
    / "metadata_with_folds.csv"
)

OUTPUT_DIR = ROOT / "results"

OVERALL_OUTPUT = (
    OUTPUT_DIR
    / "microphone_stratified_performance.csv"
)

FOLD_OUTPUT = (
    OUTPUT_DIR
    / "microphone_stratified_performance_by_fold.csv"
)

SPEAKER_OUTPUT = (
    OUTPUT_DIR
    / "microphone_stratified_performance_by_speaker.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "microphone_stratified_performance_summary.txt"
)


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "HuBERT": (
        "pred_hubert",
        "p_dys_hubert",
    ),
    "Residual HuBERT": (
        "pred_residual_hubert",
        "p_dys_residual_hubert",
    ),
}


# ============================================================
# MICROPHONE PARSING
# ============================================================

def extract_microphone(filename):

    if pd.isna(filename):
        return "unknown"

    filename = str(filename)

    match = re.search(
        r"_(arrayMic|headMic)_",
        filename,
        flags=re.IGNORECASE,
    )

    if match is None:
        return "unknown"

    value = match.group(1).lower()

    if value == "arraymic":
        return "arrayMic"

    if value == "headmic":
        return "headMic"

    return "unknown"


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    result = {
        "n": len(y_true),

        "n_healthy":
            int((y_true == 0).sum()),

        "n_dysarthria":
            int((y_true == 1).sum()),

        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
            ),
    }

    # --------------------------------------------------------
    # BA / sensitivity / specificity require both classes
    # for meaningful interpretation.
    # --------------------------------------------------------

    if len(np.unique(y_true)) == 2:

        result[
            "balanced_accuracy"
        ] = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        result[
            "sensitivity"
        ] = recall_score(
            y_true,
            y_pred,
            pos_label=1,
        )

        result[
            "specificity"
        ] = recall_score(
            y_true,
            y_pred,
            pos_label=0,
        )

        result[
            "auroc"
        ] = roc_auc_score(
            y_true,
            y_prob,
        )

    else:

        result[
            "balanced_accuracy"
        ] = np.nan

        result[
            "sensitivity"
        ] = np.nan

        result[
            "specificity"
        ] = np.nan

        result[
            "auroc"
        ] = np.nan

    return result


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("Loading data...")

    oof = pd.read_csv(
        OOF_PATH
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"OOF predictions: {len(oof)}"
    )

    print(
        f"Metadata: {len(metadata)}"
    )

    # --------------------------------------------------------
    # Validate OOF uniqueness
    # --------------------------------------------------------

    if oof[
        "dataset_index"
    ].duplicated().any():

        raise ValueError(
            "Duplicate dataset_index "
            "in OOF predictions."
        )

    # --------------------------------------------------------
    # Only take metadata columns that we need.
    #
    # This avoids accidental duplicate speaker/fold columns.
    # --------------------------------------------------------

    required_metadata = [
        "dataset_index",
        "filename",
    ]

    missing = [
        column
        for column in required_metadata
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Metadata missing: {missing}"
        )

    metadata_small = metadata[
        required_metadata
    ].copy()

    data = oof.merge(
        metadata_small,
        on="dataset_index",
        how="left",
        validate="one_to_one",
    )

    if data[
        "filename"
    ].isna().any():

        raise ValueError(
            "Some OOF recordings could not "
            "be matched to metadata."
        )

    # --------------------------------------------------------
    # Parse microphone
    # --------------------------------------------------------

    data[
        "microphone"
    ] = data[
        "filename"
    ].apply(
        extract_microphone
    )

    print("\nMicrophone counts:")

    print(
        data[
            "microphone"
        ].value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------------
    # Validate required OOF columns
    # --------------------------------------------------------

    required_oof = [
        "dataset_index",
        "speaker_id",
        "fold",
        "true_label",
    ]

    for (
        pred_column,
        prob_column
    ) in MODELS.values():

        required_oof.extend([
            pred_column,
            prob_column,
        ])

    missing = [
        column
        for column in required_oof
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OOF columns: {missing}"
        )

    return data


# ============================================================
# OVERALL MICROPHONE ANALYSIS
# ============================================================

def analyze_overall(data):

    rows = []

    valid = data[
        data["microphone"].isin(
            [
                "arrayMic",
                "headMic",
            ]
        )
    ]

    for microphone in [
        "arrayMic",
        "headMic",
    ]:

        subset = valid[
            valid["microphone"]
            == microphone
        ]

        for (
            model_name,
            (
                pred_column,
                prob_column,
            ),
        ) in MODELS.items():

            metrics = (
                calculate_metrics(
                    subset[
                        "true_label"
                    ],
                    subset[
                        pred_column
                    ],
                    subset[
                        prob_column
                    ],
                )
            )

            rows.append({
                "microphone":
                    microphone,

                "model":
                    model_name,

                **metrics,
            })

    return pd.DataFrame(rows)


# ============================================================
# FOLD × MICROPHONE
# ============================================================

def analyze_by_fold(data):

    rows = []

    valid = data[
        data["microphone"].isin(
            [
                "arrayMic",
                "headMic",
            ]
        )
    ]

    for fold in sorted(
        valid["fold"].unique()
    ):

        fold_data = valid[
            valid["fold"] == fold
        ]

        for microphone in [
            "arrayMic",
            "headMic",
        ]:

            subset = fold_data[
                fold_data["microphone"]
                == microphone
            ]

            if len(subset) == 0:
                continue

            for (
                model_name,
                (
                    pred_column,
                    prob_column,
                ),
            ) in MODELS.items():

                metrics = (
                    calculate_metrics(
                        subset[
                            "true_label"
                        ],
                        subset[
                            pred_column
                        ],
                        subset[
                            prob_column
                        ],
                    )
                )

                rows.append({
                    "fold":
                        int(fold),

                    "microphone":
                        microphone,

                    "model":
                        model_name,

                    **metrics,
                })

    return pd.DataFrame(rows)


# ============================================================
# SPEAKER × MICROPHONE
# ============================================================

def analyze_by_speaker(data):

    rows = []

    valid = data[
        data["microphone"].isin(
            [
                "arrayMic",
                "headMic",
            ]
        )
    ]

    for (
        speaker_id,
        microphone
    ), subset in valid.groupby(
        [
            "speaker_id",
            "microphone",
        ]
    ):

        # Each TORGO speaker belongs to only one class.
        # Therefore AUROC / BA are NOT meaningful here.
        true_label = int(
            subset[
                "true_label"
            ].iloc[0]
        )

        for (
            model_name,
            (
                pred_column,
                prob_column,
            ),
        ) in MODELS.items():

            accuracy = (
                subset[
                    pred_column
                ].to_numpy()
                == true_label
            ).mean()

            rows.append({
                "speaker_id":
                    speaker_id,

                "fold":
                    int(
                        subset[
                            "fold"
                        ].iloc[0]
                    ),

                "true_label":
                    true_label,

                "microphone":
                    microphone,

                "model":
                    model_name,

                "n":
                    len(subset),

                "accuracy":
                    accuracy,

                "mean_p_dys":
                    subset[
                        prob_column
                    ].mean(),

                "median_p_dys":
                    subset[
                        prob_column
                    ].median(),
            })

    return pd.DataFrame(rows)


# ============================================================
# ADD DELTAS
# ============================================================

def build_comparison_table(
    results,
    grouping_columns,
):

    hubert = results[
        results["model"]
        == "HuBERT"
    ].copy()

    residual = results[
        results["model"]
        == "Residual HuBERT"
    ].copy()

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auroc",
        "sensitivity",
        "specificity",
    ]

    keep_hubert = (
        grouping_columns
        + ["n", "n_healthy", "n_dysarthria"]
        + metric_columns
    )

    keep_residual = (
        grouping_columns
        + metric_columns
    )

    hubert = hubert[
        keep_hubert
    ]

    residual = residual[
        keep_residual
    ]

    merged = hubert.merge(
        residual,
        on=grouping_columns,
        suffixes=(
            "_hubert",
            "_residual",
        ),
        validate="one_to_one",
    )

    for metric in metric_columns:

        merged[
            f"delta_{metric}"
        ] = (
            merged[
                f"{metric}_residual"
            ]
            - merged[
                f"{metric}_hubert"
            ]
        )

    return merged


# ============================================================
# PRINT
# ============================================================

def print_overall(comparison):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OVERALL MICROPHONE-STRATIFIED PERFORMANCE"
    )

    print(
        "=" * 80
    )

    columns = [
        "microphone",
        "n",
        "n_healthy",
        "n_dysarthria",

        "balanced_accuracy_hubert",
        "balanced_accuracy_residual",
        "delta_balanced_accuracy",

        "auroc_hubert",
        "auroc_residual",
        "delta_auroc",
    ]

    print(
        comparison[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


def print_fold_results(comparison):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FOLD × MICROPHONE"
    )

    print(
        "=" * 80
    )

    columns = [
        "fold",
        "microphone",
        "n",

        "balanced_accuracy_hubert",
        "balanced_accuracy_residual",
        "delta_balanced_accuracy",

        "auroc_hubert",
        "auroc_residual",
        "delta_auroc",
    ]

    print(
        comparison[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

def save_summary(
    overall_comparison,
    fold_comparison,
):

    lines = []

    lines.append(
        "Microphone-Stratified OOF Evaluation"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        "Overall"
    )

    lines.append(
        "-" * 80
    )

    for _, row in (
        overall_comparison.iterrows()
    ):

        lines.append(
            f"{row['microphone']}"
        )

        lines.append(
            f"  N: {int(row['n'])}"
        )

        lines.append(
            "  HuBERT BA: "
            f"{row['balanced_accuracy_hubert']:.3f}"
        )

        lines.append(
            "  Residual BA: "
            f"{row['balanced_accuracy_residual']:.3f}"
        )

        lines.append(
            "  Delta BA: "
            f"{row['delta_balanced_accuracy']:+.3f}"
        )

        lines.append(
            "  HuBERT AUROC: "
            f"{row['auroc_hubert']:.3f}"
        )

        lines.append(
            "  Residual AUROC: "
            f"{row['auroc_residual']:.3f}"
        )

        lines.append(
            "  Delta AUROC: "
            f"{row['delta_auroc']:+.3f}"
        )

        lines.append("")

    lines.append(
        "Fold x microphone"
    )

    lines.append(
        "-" * 80
    )

    for _, row in (
        fold_comparison.iterrows()
    ):

        lines.append(
            f"Fold {int(row['fold'])} "
            f"| {row['microphone']} "
            f"| N={int(row['n'])}"
        )

        lines.append(
            "  BA: "
            f"{row['balanced_accuracy_hubert']:.3f}"
            " -> "
            f"{row['balanced_accuracy_residual']:.3f}"
            " "
            f"({row['delta_balanced_accuracy']:+.3f})"
        )

        lines.append(
            "  AUROC: "
            f"{row['auroc_hubert']:.3f}"
            " -> "
            f"{row['auroc_residual']:.3f}"
            " "
            f"({row['delta_auroc']:+.3f})"
        )

        lines.append("")

    SUMMARY_OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = analyze_overall(
        data
    )

    overall_comparison = (
        build_comparison_table(
            overall,
            grouping_columns=[
                "microphone"
            ],
        )
    )

    # --------------------------------------------------------
    # Fold × microphone
    # --------------------------------------------------------

    fold_results = analyze_by_fold(
        data
    )

    fold_comparison = (
        build_comparison_table(
            fold_results,
            grouping_columns=[
                "fold",
                "microphone",
            ],
        )
    )

    # --------------------------------------------------------
    # Speaker × microphone
    # --------------------------------------------------------

    speaker_results = (
        analyze_by_speaker(
            data
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    overall_comparison.to_csv(
        OVERALL_OUTPUT,
        index=False,
    )

    fold_comparison.to_csv(
        FOLD_OUTPUT,
        index=False,
    )

    speaker_results.to_csv(
        SPEAKER_OUTPUT,
        index=False,
    )

    save_summary(
        overall_comparison,
        fold_comparison,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_overall(
        overall_comparison
    )

    print_fold_results(
        fold_comparison
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SPEAKER × MICROPHONE RESULTS SAVED"
    )

    print(
        "=" * 80
    )

    print(
        SPEAKER_OUTPUT
    )

    print("\nSaved:")
    print(OVERALL_OUTPUT)
    print(FOLD_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()