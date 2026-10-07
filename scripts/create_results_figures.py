from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ------------------------------------------------------------
# Master results
# ------------------------------------------------------------

MASTER_TABLE_PATH = (
    RESULTS_DIR
    / "master_results_table.csv"
)


# ------------------------------------------------------------
# Figure 1
# ------------------------------------------------------------

FIGURE_1_PATH = (
    FIGURES_DIR
    / "figure_1_beyond_wer.png"
)

FIGURE_1_PDF_PATH = (
    FIGURES_DIR
    / "figure_1_beyond_wer.pdf"
)


# ------------------------------------------------------------
# Figure 2
# ------------------------------------------------------------

RESIDUAL_ABLATION_PATH = (
    RESULTS_DIR
    / "residual_group_ablation_fold_results.csv"
)

FIGURE_2_PATH = (
    FIGURES_DIR
    / "figure_2_residual_ablation.png"
)

FIGURE_2_PDF_PATH = (
    FIGURES_DIR
    / "figure_2_residual_ablation.pdf"
)


# ------------------------------------------------------------
# Figure 3
# ------------------------------------------------------------

PREDICTION_FLIPS_BY_SPEAKER_PATH = (
    RESULTS_DIR
    / "prediction_flips_by_speaker.csv"
)

FIGURE_3_PATH = (
    FIGURES_DIR
    / "figure_3_speaker_residualization_effects.png"
)

FIGURE_3_PDF_PATH = (
    FIGURES_DIR
    / "figure_3_speaker_residualization_effects.pdf"
)


# ------------------------------------------------------------
# Figure 4
# ------------------------------------------------------------

ACOUSTIC_PROBE_SUMMARY_PATH = (
    RESULTS_DIR
    / "hubert_acoustic_probe_summary.csv"
)

FIGURE_4_PATH = (
    FIGURES_DIR
    / "figure_4_acoustic_probe.png"
)

FIGURE_4_PDF_PATH = (
    FIGURES_DIR
    / "figure_4_acoustic_probe.pdf"
)


# ------------------------------------------------------------
# Figure 5
# ------------------------------------------------------------

MICROPHONE_BY_FOLD_PATH = (
    RESULTS_DIR
    / "microphone_stratified_performance_by_fold.csv"
)

FIGURE_5_PATH = (
    FIGURES_DIR
    / "figure_5_microphone_stratification.png"
)

FIGURE_5_PDF_PATH = (
    FIGURES_DIR
    / "figure_5_microphone_stratification.pdf"
)


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

EXPERIMENTS = [
    {
        "representation": "Acoustic",
        "condition": "Full TORGO",
        "path": (
            RESULTS_DIR
            / "acoustic_baseline_results.csv"
        ),
        "n": 16552,
        "n_prompts": np.nan,
    },
    {
        "representation": "Acoustic",
        "condition": "Perfect ASR",
        "path": (
            RESULTS_DIR
            / "perfect_asr_acoustic_fold_results.csv"
        ),
        "n": 10121,
        "n_prompts": np.nan,
    },
    {
        "representation": "Acoustic",
        "condition": "Strict shared-prompt",
        "path": (
            RESULTS_DIR
            / "strict_shared_prompt_acoustic_fold_results.csv"
        ),
        "n": 4962,
        "n_prompts": 213,
    },
    {
        "representation": "HuBERT",
        "condition": "Full TORGO",
        "path": (
            RESULTS_DIR
            / "hubert_baseline_fold_results.csv"
        ),
        "n": 16552,
        "n_prompts": np.nan,
    },
    {
        "representation": "HuBERT",
        "condition": "Perfect ASR",
        "path": (
            RESULTS_DIR
            / "hubert_perfect_asr_fold_results.csv"
        ),
        "n": 10121,
        "n_prompts": np.nan,
    },
    {
        "representation": "HuBERT",
        "condition": "Strict shared-prompt",
        "path": (
            RESULTS_DIR
            / "hubert_strict_shared_prompt_fold_results.csv"
        ),
        "n": 4962,
        "n_prompts": 213,
    },
]


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    """
    Fail early with a readable error if a required result file
    is missing.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )


def require_columns(
    df,
    columns,
    source_name,
):
    """
    Validate that all required columns exist.
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {source_name}: "
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


def find_metric_column(
    df,
    candidates,
):
    """
    Return the first matching column name.

    This handles small naming differences such as:
        auroc
        roc_auc

    or:
        r2_mean
        mean_r2
    """

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not find any of these columns:\n"
        f"{candidates}\n\n"
        f"Available columns:\n"
        f"{list(df.columns)}"
    )


def save_figure(
    fig,
    png_path,
    pdf_path,
):
    """
    Save each figure as:
        - high-resolution PNG for GitHub / README
        - vector PDF for the research report
    """

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MASTER RESULTS TABLE
# ============================================================

def build_master_table():

    rows = []

    for experiment in EXPERIMENTS:

        path = experiment["path"]

        require_file(path)

        df = pd.read_csv(path)

        print(
            "\n"
            + "=" * 70
        )

        print(
            experiment["representation"],
            "|",
            experiment["condition"],
        )

        print(
            "=" * 70
        )

        print(
            "File:",
            path.name,
        )

        print(
            "Rows:",
            len(df),
        )

        print(
            "Columns:",
            list(df.columns),
        )

        # ----------------------------------------------------
        # Identify metric columns
        # ----------------------------------------------------

        ba_column = find_metric_column(
            df,
            [
                "balanced_accuracy",
                "balanced_acc",
                "ba",
            ],
        )

        f1_column = find_metric_column(
            df,
            [
                "macro_f1",
                "f1_macro",
                "f1",
            ],
        )

        auroc_column = find_metric_column(
            df,
            [
                "auroc",
                "roc_auc",
                "auc",
            ],
        )

        # ----------------------------------------------------
        # Expected five speaker-independent folds
        # ----------------------------------------------------

        if len(df) != 5:

            print(
                "WARNING: Expected 5 fold rows, "
                f"found {len(df)}."
            )

        # ----------------------------------------------------
        # Mean ± sample SD across folds
        # ----------------------------------------------------

        row = {
            "representation":
                experiment[
                    "representation"
                ],

            "condition":
                experiment[
                    "condition"
                ],

            "n":
                experiment[
                    "n"
                ],

            "n_prompts":
                experiment[
                    "n_prompts"
                ],

            "n_folds":
                len(df),

            "balanced_accuracy_mean":
                df[
                    ba_column
                ].mean(),

            "balanced_accuracy_std":
                df[
                    ba_column
                ].std(
                    ddof=1
                ),

            "macro_f1_mean":
                df[
                    f1_column
                ].mean(),

            "macro_f1_std":
                df[
                    f1_column
                ].std(
                    ddof=1
                ),

            "auroc_mean":
                df[
                    auroc_column
                ].mean(),

            "auroc_std":
                df[
                    auroc_column
                ].std(
                    ddof=1
                ),
        }

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def print_master_table(
    master,
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MASTER RESULTS TABLE"
    )

    print(
        "=" * 90
    )

    columns = [
        "representation",
        "condition",
        "n",
        "n_prompts",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "auroc_mean",
        "auroc_std",
    ]

    print(
        master[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


# ============================================================
# FIGURE 1
# BEYOND-WER
# ============================================================

def create_beyond_wer_figure(
    master,
):

    conditions = [
        "Full TORGO",
        "Perfect ASR",
        "Strict shared-prompt",
    ]

    representations = [
        "Acoustic",
        "HuBERT",
    ]

    fig, ax = plt.subplots(
        figsize=(
            9,
            5.5,
        )
    )

    x = np.arange(
        len(conditions)
    )

    offsets = {
        "Acoustic": -0.08,
        "HuBERT": 0.08,
    }

    markers = {
        "Acoustic": "o",
        "HuBERT": "s",
    }

    for representation in representations:

        subset = (
            master[
                master[
                    "representation"
                ]
                == representation
            ]
            .set_index(
                "condition"
            )
            .loc[
                conditions
            ]
        )

        means = (
            subset[
                "auroc_mean"
            ]
            .to_numpy()
        )

        stds = (
            subset[
                "auroc_std"
            ]
            .to_numpy()
        )

        x_values = (
            x
            + offsets[
                representation
            ]
        )

        ax.errorbar(
            x_values,
            means,
            yerr=stds,
            marker=markers[
                representation
            ],
            linewidth=2,
            markersize=7,
            capsize=3,
            elinewidth=1.3,
            alpha=0.75,
            label=representation,
        )

        # ----------------------------------------------------
        # Mean value labels
        # ----------------------------------------------------

        for (
            x_value,
            mean,
        ) in zip(
            x_values,
            means,
        ):

            ax.annotate(
                f"{mean:.3f}",
                (
                    x_value,
                    mean,
                ),
                xytext=(
                    0,
                    9,
                ),
                textcoords=(
                    "offset points"
                ),
                ha="center",
                fontsize=9,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels([
        "Full TORGO\nN = 16,552",
        "Perfect ASR\nN = 10,121",
        (
            "Strict shared-prompt\n"
            "N = 4,962 | 213 prompts"
        ),
    ])

    ax.set_ylabel(
        "AUROC"
    )

    ax.set_xlabel(
        "Evaluation condition"
    )

    ax.set_title(
        "Speech-status separability under "
        "increasingly strict transcript controls"
    )

    ax.set_ylim(
        0.50,
        1.02,
    )

    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    save_figure(
        fig,
        FIGURE_1_PATH,
        FIGURE_1_PDF_PATH,
    )


# ============================================================
# FIGURE 2
# RESIDUAL GROUP ABLATION
# ============================================================

def create_residual_ablation_figure():

    require_file(
        RESIDUAL_ABLATION_PATH
    )

    df = pd.read_csv(
        RESIDUAL_ABLATION_PATH
    )

    require_columns(
        df,
        [
            "condition",
            "auroc",
        ],
        RESIDUAL_ABLATION_PATH.name,
    )

    conditions = [
        "hubert_original",
        "residual_duration_only",
        "residual_prosody",
        "residual_timing",
        "residual_energy",
        "residual_all_acoustic",
    ]

    labels = {
        "hubert_original":
            "Original HuBERT",

        "residual_duration_only":
            "Residualized: duration",

        "residual_prosody":
            "Residualized: prosody",

        "residual_timing":
            "Residualized: timing",

        "residual_energy":
            "Residualized: energy",

        "residual_all_acoustic":
            (
                "Residualized: "
                "all measured acoustics"
            ),
    }

    # --------------------------------------------------------
    # Validate conditions
    # --------------------------------------------------------

    available_conditions = set(
        df[
            "condition"
        ].unique()
    )

    missing_conditions = [
        condition
        for condition in conditions
        if condition
        not in available_conditions
    ]

    if missing_conditions:

        raise ValueError(
            "Missing residual conditions:\n"
            f"{missing_conditions}\n\n"
            "Available conditions:\n"
            f"{sorted(available_conditions)}"
        )

    # --------------------------------------------------------
    # Aggregate across folds
    # --------------------------------------------------------

    summary = (
        df
        .groupby(
            "condition"
        )
        .agg(
            auroc_mean=(
                "auroc",
                "mean",
            ),
            auroc_std=(
                "auroc",
                "std",
            ),
        )
        .reindex(
            conditions
        )
    )

    baseline = summary.loc[
        "hubert_original",
        "auroc_mean",
    ]

    summary[
        "delta_auroc"
    ] = (
        summary[
            "auroc_mean"
        ]
        - baseline
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESIDUAL ABLATION"
    )

    print(
        "=" * 70
    )

    print(
        summary.to_string(
            float_format=lambda x:
                f"{x:.3f}",
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            9,
            5.8,
        )
    )

    y = np.arange(
        len(conditions)
    )

    means = (
        summary[
            "auroc_mean"
        ]
        .to_numpy()
    )

    stds = (
        summary[
            "auroc_std"
        ]
        .to_numpy()
    )

    ax.errorbar(
        means,
        y,
        xerr=stds,
        fmt="o",
        markersize=8,
        capsize=3,
        elinewidth=1.1,
        alpha=0.7,
    )

    # Original HuBERT reference
    ax.axvline(
        baseline,
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels([
        labels[
            condition
        ]
        for condition
        in conditions
    ])

    ax.invert_yaxis()

    # --------------------------------------------------------
    # Mean and delta annotations
    # --------------------------------------------------------

    for (
        i,
        condition,
    ) in enumerate(
        conditions
    ):

        mean = summary.loc[
            condition,
            "auroc_mean",
        ]

        delta = summary.loc[
            condition,
            "delta_auroc",
        ]

        if (
            condition
            == "hubert_original"
        ):

            text = (
                f"{mean:.3f}"
            )

        else:

            text = (
                f"{mean:.3f}  "
                f"(Δ {delta:+.3f})"
            )

        ax.annotate(
            text,
            (
                mean,
                i,
            ),
            xytext=(
                10,
                0,
            ),
            textcoords=(
                "offset points"
            ),
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(
        "AUROC"
    )

    ax.set_title(
        "HuBERT separability after "
        "acoustic-feature residualization"
    )

    ax.set_xlim(
        0.35,
        1.02,
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    save_figure(
        fig,
        FIGURE_2_PATH,
        FIGURE_2_PDF_PATH,
    )


# ============================================================
# FIGURE 3
# SPEAKER-SPECIFIC RESIDUALIZATION EFFECTS
# ============================================================

def create_speaker_residualization_figure():

    require_file(
        PREDICTION_FLIPS_BY_SPEAKER_PATH
    )

    df = pd.read_csv(
        PREDICTION_FLIPS_BY_SPEAKER_PATH
    )

    require_columns(
        df,
        [
            "speaker_id",
            "speech_status",
            "mean_delta_p_dys",
        ],
        PREDICTION_FLIPS_BY_SPEAKER_PATH.name,
    )

    # --------------------------------------------------------
    # Validate one row per speaker
    # --------------------------------------------------------

    if (
        df[
            "speaker_id"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "prediction_flips_by_speaker.csv "
            "contains duplicate speaker IDs."
        )

    # --------------------------------------------------------
    # Normalize status labels for plotting
    # --------------------------------------------------------

    df[
        "speech_status"
    ] = (
        df[
            "speech_status"
        ]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    unexpected_status = set(
        df[
            "speech_status"
        ].unique()
    ) - {
        "healthy",
        "dysarthria",
    }

    if unexpected_status:

        raise ValueError(
            "Unexpected speech_status values:\n"
            f"{sorted(unexpected_status)}"
        )

    # --------------------------------------------------------
    # Sort from strongest shift toward healthy
    # to strongest shift toward dysarthria
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "mean_delta_p_dys"
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(df)
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            7,
        )
    )

    # --------------------------------------------------------
    # Lines from zero to each speaker
    # --------------------------------------------------------

    for i, row in df.iterrows():
        ax.hlines(
            y=i,
            xmin=min(0, row["mean_delta_p_dys"]),
            xmax=max(0, row["mean_delta_p_dys"]),
            linewidth=1.4,
            alpha=0.35,
        )

    # --------------------------------------------------------
    # Plot healthy and dysarthria separately
    # --------------------------------------------------------

    markers = {
        "healthy": "o",
        "dysarthria": "s",
    }

    labels = {
        "healthy": "Healthy",
        "dysarthria": "Dysarthria",
    }

    for status in [
        "healthy",
        "dysarthria",
    ]:

        subset = df[
            df[
                "speech_status"
            ]
            == status
        ]

        ax.scatter(
            subset[
                "mean_delta_p_dys"
            ],
            subset.index,
            marker=markers[
                status
            ],
            s=70,
            label=labels[
                status
            ],
            zorder=3,
        )

    # --------------------------------------------------------
    # Zero reference
    # --------------------------------------------------------

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )

    # --------------------------------------------------------
    # Speaker labels
    # --------------------------------------------------------

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        df[
            "speaker_id"
        ]
    )

    # --------------------------------------------------------
    # Direction labels
    # --------------------------------------------------------

    ax.text(
        0.02,
        1.02,
        "← toward healthy",
        transform=ax.transAxes,
        ha="left",
        fontsize=10,
    )

    ax.text(
        0.98,
        1.02,
        "toward dysarthria →",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
    )

    ax.set_xlabel(
        "Mean ΔP(dysarthria) after acoustic residualization"
    )

    ax.set_ylabel(
        "Speaker"
    )

    ax.set_title(
        "Acoustic residualization has "
        "speaker-dependent effects"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    ax.legend(
        frameon=False,
    )

    save_figure(
        fig,
        FIGURE_3_PATH,
        FIGURE_3_PDF_PATH,
    )

    # --------------------------------------------------------
    # Print useful diagnostic table
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SPEAKER RESIDUALIZATION EFFECTS"
    )

    print(
        "=" * 70
    )

    print(
        df[
            [
                "speaker_id",
                "speech_status",
                "mean_delta_p_dys",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


# ============================================================
# FIGURE 4
# ACOUSTIC INFORMATION ACCESSIBLE FROM HUBERT
# ============================================================

def create_acoustic_probe_figure():

    require_file(
        ACOUSTIC_PROBE_SUMMARY_PATH
    )

    df = pd.read_csv(
        ACOUSTIC_PROBE_SUMMARY_PATH
    )

    print(
        "\nAcoustic probe columns:",
        list(df.columns),
    )

    # --------------------------------------------------------
    # Identify columns robustly
    # --------------------------------------------------------

    feature_column = find_metric_column(
        df,
        [
            "feature",
            "target",
            "acoustic_feature",
        ],
    )

    r2_mean_column = find_metric_column(
        df,
        [
            "r2_mean",
            "mean_r2",
            "r2_mean_mean",
        ],
    )

    r2_std_column = find_metric_column(
        df,
        [
            "r2_std",
            "std_r2",
            "r2_mean_std",
        ],
    )

    # --------------------------------------------------------
    # speech_ratio and pause_ratio are near-complements.
    # Keep pause_ratio only to avoid redundant visualization.
    # --------------------------------------------------------

    df = df[
        df[
            feature_column
        ]
        != "speech_ratio"
    ].copy()

    labels = {
        "audio_duration":
            "Audio duration",

        "voiced_duration":
            "Voiced duration",

        "pause_ratio":
            "Pause ratio",

        "number_of_pauses":
            "Number of pauses",

        "mean_pause_duration":
            "Mean pause duration",

        "f0_mean":
            "F0 mean",

        "f0_std":
            "F0 variability",

        "rms_mean":
            "RMS mean",

        "rms_std":
            "RMS variability",
    }

    df[
        "display_name"
    ] = (
        df[
            feature_column
        ]
        .map(
            labels
        )
        .fillna(
            df[
                feature_column
            ]
        )
    )

    # --------------------------------------------------------
    # Sort by R²
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            r2_mean_column
        )
        .reset_index(
            drop=True
        )
    )

    y = np.arange(
        len(df)
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    ax.errorbar(
        df[
            r2_mean_column
        ],
        y,
        xerr=df[
            r2_std_column
        ],
        fmt="o",
        markersize=7,
        capsize=3,
        elinewidth=1.1,
        alpha=0.75,
    )

    # --------------------------------------------------------
    # R² = 0 reference
    # --------------------------------------------------------

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        df[
            "display_name"
        ]
    )

    ax.set_xlabel(
        "Cross-speaker $R^2$"
    )

    ax.set_title(
        "Acoustic characteristics linearly accessible "
        "from frozen HuBERT representations"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    # --------------------------------------------------------
    # Mean value annotations
    # --------------------------------------------------------

    for i, value in enumerate(
        df[
            r2_mean_column
        ]
    ):

        ax.annotate(
            f"{value:.3f}",
            (
                value,
                i,
            ),
            xytext=(
                8,
                0,
            ),
            textcoords=(
                "offset points"
            ),
            va="center",
            fontsize=8,
        )

    save_figure(
        fig,
        FIGURE_4_PATH,
        FIGURE_4_PDF_PATH,
    )


# ============================================================
# FIGURE 5
# MICROPHONE-STRATIFIED PERFORMANCE
# ============================================================

def create_microphone_figure():

    require_file(
        MICROPHONE_BY_FOLD_PATH
    )

    df = pd.read_csv(
        MICROPHONE_BY_FOLD_PATH
    )

    require_columns(
        df,
        [
            "fold",
            "microphone",
            "auroc_hubert",
            "auroc_residual",
        ],
        MICROPHONE_BY_FOLD_PATH.name,
    )

    microphones = [
        "arrayMic",
        "headMic",
    ]

    available_microphones = set(
        df["microphone"].unique()
    )

    missing_microphones = [
        microphone
        for microphone in microphones
        if microphone not in available_microphones
    ]

    if missing_microphones:
        raise ValueError(
            "Missing microphone categories:\n"
            f"{missing_microphones}\n\n"
            "Available:\n"
            f"{sorted(available_microphones)}"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    # Color identifies microphone.
    # Do not specify actual colors manually:
    # use matplotlib's default color cycle.
    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    microphone_styles = {
        "arrayMic": {
            "label": "Array microphone",
            "marker": "o",
            "color": default_colors[0],
        },
        "headMic": {
            "label": "Head microphone",
            "marker": "s",
            "color": default_colors[1],
        },
    }

    # --------------------------------------------------------
    # Plot each microphone:
    # same color, different line style for representation
    # --------------------------------------------------------

    for microphone in microphones:

        subset = (
            df[
                df["microphone"]
                == microphone
            ]
            .sort_values("fold")
        )

        style = microphone_styles[
            microphone
        ]

        # Original HuBERT
        ax.plot(
            subset["fold"],
            subset["auroc_hubert"],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.2,
            markersize=7,
        )

        # Residualized HuBERT
        ax.plot(
            subset["fold"],
            subset["auroc_residual"],
            color=style["color"],
            marker=style["marker"],
            linestyle="--",
            linewidth=1.8,
            markersize=7,
            alpha=0.75,
        )

    # --------------------------------------------------------
    # Chance level
    # --------------------------------------------------------

    ax.axhline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        alpha=0.6,
    )

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    folds = sorted(
        df["fold"].unique()
    )

    ax.set_xticks(
        folds
    )

    ax.set_xlabel(
        "Speaker-independent fold"
    )

    ax.set_ylabel(
        "AUROC"
    )

    ax.set_ylim(
        0.15,
        1.02,
    )

    ax.set_title(
        "HuBERT performance across speaker folds "
        "and microphone conditions"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    # --------------------------------------------------------
    # Two separate legends
    # --------------------------------------------------------

    from matplotlib.lines import Line2D

    microphone_handles = [
        Line2D(
            [0],
            [0],
            color=microphone_styles[
                "arrayMic"
            ]["color"],
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Array microphone",
        ),
        Line2D(
            [0],
            [0],
            color=microphone_styles[
                "headMic"
            ]["color"],
            marker="s",
            linestyle="-",
            linewidth=2,
            label="Head microphone",
        ),
    ]

    representation_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2,
            label="Original HuBERT",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
            label="Residualized HuBERT",
        ),
    ]

    microphone_legend = ax.legend(
        handles=microphone_handles,
        title="Microphone",
        frameon=False,
        loc="upper right",
    )

    ax.add_artist(
        microphone_legend
    )

    ax.legend(
        handles=representation_handles,
        title="Representation",
        frameon=False,
        loc="lower left",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_figure(
        fig,
        FIGURE_5_PATH,
        FIGURE_5_PDF_PATH,
    )

# ============================================================
# MAIN
# ============================================================

def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Master table
    # --------------------------------------------------------

    master = (
        build_master_table()
    )

    master.to_csv(
        MASTER_TABLE_PATH,
        index=False,
    )

    print_master_table(
        master
    )

    # --------------------------------------------------------
    # Generate all final figures
    # --------------------------------------------------------

    print(
        "\nGenerating Figure 1..."
    )

    create_beyond_wer_figure(
        master
    )

    print(
        "Generating Figure 2..."
    )

    create_residual_ablation_figure()

    print(
        "Generating Figure 3..."
    )

    create_speaker_residualization_figure()

    print(
        "Generating Figure 4..."
    )

    create_acoustic_probe_figure()

    print(
        "Generating Figure 5..."
    )

    create_microphone_figure()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ALL FIGURES GENERATED SUCCESSFULLY"
    )

    print(
        "=" * 90
    )

    print(
        "\nMaster table:"
    )

    print(
        MASTER_TABLE_PATH
    )

    print(
        "\nFigures:"
    )

    figure_paths = [
        FIGURE_1_PATH,
        FIGURE_1_PDF_PATH,

        FIGURE_2_PATH,
        FIGURE_2_PDF_PATH,

        FIGURE_3_PATH,
        FIGURE_3_PDF_PATH,

        FIGURE_4_PATH,
        FIGURE_4_PDF_PATH,

        FIGURE_5_PATH,
        FIGURE_5_PDF_PATH,
    ]

    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()