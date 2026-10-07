from pathlib import Path

import numpy as np
import pandas as pd


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

ACOUSTIC_PATH = (
    ROOT
    / "data"
    / "acoustic_features.csv"
)

WHISPER_PATH = (
    ROOT
    / "data"
    / "whisper_full_predictions.csv"
)

RESULTS_DIR = ROOT / "results"

SPEAKER_OUTPUT = (
    RESULTS_DIR
    / "prediction_error_by_speaker.csv"
)

PROMPT_OUTPUT = (
    RESULTS_DIR
    / "prediction_error_by_prompt.csv"
)

ERROR_OUTPUT = (
    RESULTS_DIR
    / "prediction_error_recordings.csv"
)

TOP_ERRORS_OUTPUT = (
    RESULTS_DIR
    / "prediction_error_top_confident.csv"
)

FLIPS_OUTPUT = (
    RESULTS_DIR
    / "prediction_flips_hubert_residual.csv"
)

FLIP_SPEAKER_OUTPUT = (
    RESULTS_DIR
    / "prediction_flips_by_speaker.csv"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_CONFIG = {
    "acoustic": {
        "pred": "pred_acoustic",
        "prob": "p_dys_acoustic",
    },
    "hubert": {
        "pred": "pred_hubert",
        "prob": "p_dys_hubert",
    },
    "hubert_acoustic": {
        "pred": "pred_hubert_acoustic",
        "prob": "p_dys_hubert_acoustic",
    },
    "residual_hubert": {
        "pred": "pred_residual_hubert",
        "prob": "p_dys_residual_hubert",
    },
}

TOP_N = 30


# ============================================================
# LOAD + MERGE
# ============================================================

def load_data():

    print("=" * 70)
    print("Loading prediction data")
    print("=" * 70)

    oof = pd.read_csv(OOF_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    acoustic = pd.read_csv(ACOUSTIC_PATH)
    whisper = pd.read_csv(WHISPER_PATH)

    print(f"OOF predictions: {len(oof)}")
    print(f"Metadata:        {len(metadata)}")
    print(f"Acoustic:        {len(acoustic)}")
    print(f"Whisper:         {len(whisper)}")

    # --------------------------------------------------------
    # Only add metadata columns that are not already in OOF
    # --------------------------------------------------------

    metadata_columns = [
        column
        for column in [
            "dataset_index",
            "transcription",
            "filename",
            "gender",
            "duration",
        ]
        if column in metadata.columns
    ]

    metadata_extra = [
        column
        for column in metadata_columns
        if column == "dataset_index"
        or column not in oof.columns
    ]

    data = oof.merge(
        metadata[metadata_extra],
        on="dataset_index",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Acoustic features
    # --------------------------------------------------------

    acoustic_columns = [
        "dataset_index",
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

    acoustic_columns = [
        column
        for column in acoustic_columns
        if column in acoustic.columns
    ]

    data = data.merge(
        acoustic[acoustic_columns],
        on="dataset_index",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Whisper diagnostics
    # --------------------------------------------------------

    whisper_columns = [
        "dataset_index",
        "reference",
        "prediction",
        "reference_normalized",
        "prediction_normalized",
        "wer",
        "cer",
        "possible_hallucination",
        "used_chunking",
    ]

    whisper_columns = [
        column
        for column in whisper_columns
        if column in whisper.columns
    ]

    data = data.merge(
        whisper[whisper_columns],
        on="dataset_index",
        how="left",
        validate="one_to_one",
    )

    if len(data) != len(oof):
        raise ValueError(
            "Merge changed number of OOF samples."
        )

    if data["dataset_index"].duplicated().any():
        raise ValueError(
            "Duplicate dataset_index after merge."
        )

    print(f"Merged samples:  {len(data)}")
    print(f"Speakers:        {data['speaker_id'].nunique()}")

    return data


# ============================================================
# ADD ERROR INFORMATION
# ============================================================

def add_error_columns(data):

    data = data.copy()

    for model_name, config in MODEL_CONFIG.items():

        pred_column = config["pred"]
        prob_column = config["prob"]

        # ----------------------------------------------------
        # Correct / incorrect
        # ----------------------------------------------------

        data[
            f"{model_name}_correct"
        ] = (
            data[pred_column]
            == data["true_label"]
        )

        # ----------------------------------------------------
        # Probability assigned to WRONG class
        #
        # healthy:
        #   wrong-class confidence = p(dys)
        #
        # dysarthria:
        #   wrong-class confidence = 1 - p(dys)
        # ----------------------------------------------------

        data[
            f"{model_name}_wrong_confidence"
        ] = np.where(
            data["true_label"] == 0,
            data[prob_column],
            1.0 - data[prob_column],
        )

        # ----------------------------------------------------
        # Prediction margin from decision boundary
        # ----------------------------------------------------

        data[
            f"{model_name}_margin"
        ] = (
            np.abs(
                data[prob_column] - 0.5
            )
        )

    return data


# ============================================================
# SPEAKER ANALYSIS
# ============================================================

def analyze_speakers(data):

    rows = []

    for speaker_id, group in data.groupby(
        "speaker_id"
    ):

        base = {
            "speaker_id": speaker_id,
            "speech_status": (
                group["speech_status"].iloc[0]
            ),
            "fold": int(
                group["fold"].iloc[0]
            ),
            "n_recordings": len(group),
        }

        if "gender" in group.columns:
            base["gender"] = (
                group["gender"].iloc[0]
            )

        if "wer" in group.columns:
            base["mean_wer"] = (
                group["wer"].mean()
            )

            base["median_wer"] = (
                group["wer"].median()
            )

        if "audio_duration" in group.columns:
            base["mean_audio_duration"] = (
                group["audio_duration"].mean()
            )

        if "pause_ratio" in group.columns:
            base["mean_pause_ratio"] = (
                group["pause_ratio"].mean()
            )

        if "f0_mean" in group.columns:
            base["mean_f0"] = (
                group["f0_mean"].mean()
            )

        for model_name, config in MODEL_CONFIG.items():

            pred_column = config["pred"]
            prob_column = config["prob"]

            base[
                f"{model_name}_accuracy"
            ] = (
                group[pred_column]
                .eq(group["true_label"])
                .mean()
            )

            base[
                f"{model_name}_mean_p_dys"
            ] = (
                group[prob_column].mean()
            )

            base[
                f"{model_name}_median_p_dys"
            ] = (
                group[prob_column].median()
            )

        rows.append(base)

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "hubert_accuracy",
        ascending=True,
    )

    return result


# ============================================================
# PROMPT ANALYSIS
# ============================================================

def analyze_prompts(data):

    if "reference_normalized" not in data.columns:
        raise ValueError(
            "reference_normalized not available."
        )

    prompt_data = data[
        data["reference_normalized"].notna()
        & (
            data["reference_normalized"]
            .astype(str)
            .str.len()
            > 0
        )
    ].copy()

    rows = []

    for prompt, group in prompt_data.groupby(
        "reference_normalized"
    ):

        healthy_n = int(
            (group["true_label"] == 0).sum()
        )

        dys_n = int(
            (group["true_label"] == 1).sum()
        )

        healthy_speakers = (
            group.loc[
                group["true_label"] == 0,
                "speaker_id",
            ]
            .nunique()
        )

        dys_speakers = (
            group.loc[
                group["true_label"] == 1,
                "speaker_id",
            ]
            .nunique()
        )

        row = {
            "prompt": prompt,
            "n_recordings": len(group),
            "healthy_n": healthy_n,
            "dysarthria_n": dys_n,
            "healthy_speakers": healthy_speakers,
            "dysarthria_speakers": dys_speakers,
        }

        for model_name, config in MODEL_CONFIG.items():

            row[
                f"{model_name}_accuracy"
            ] = (
                group[config["pred"]]
                .eq(group["true_label"])
                .mean()
            )

            row[
                f"{model_name}_mean_p_dys"
            ] = (
                group[config["prob"]].mean()
            )

        rows.append(row)

    result = pd.DataFrame(rows)

    # Keep prompts with some meaningful coverage.
    result = result[
        result["n_recordings"] >= 10
    ].copy()

    result = result.sort_values(
        [
            "hubert_accuracy",
            "n_recordings",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return result


# ============================================================
# RECORDING-LEVEL ERRORS
# ============================================================

def analyze_recording_errors(data):

    error_rows = []

    for model_name, config in MODEL_CONFIG.items():

        incorrect = data[
            ~data[
                f"{model_name}_correct"
            ]
        ].copy()

        incorrect["model"] = model_name

        incorrect["model_probability"] = (
            incorrect[config["prob"]]
        )

        incorrect["wrong_confidence"] = (
            incorrect[
                f"{model_name}_wrong_confidence"
            ]
        )

        error_rows.append(
            incorrect
        )

    result = pd.concat(
        error_rows,
        ignore_index=True,
    )

    result = result.sort_values(
        "wrong_confidence",
        ascending=False,
    )

    return result


# ============================================================
# TOP CONFIDENT ERRORS
# ============================================================

def get_top_confident_errors(
    error_data,
):

    selected = []

    for model_name in MODEL_CONFIG:

        model_errors = error_data[
            error_data["model"]
            == model_name
        ]

        # ---------------------------------------------
        # False positives:
        # healthy predicted dysarthria
        # ---------------------------------------------

        false_positive = (
            model_errors[
                model_errors["true_label"] == 0
            ]
            .nlargest(
                TOP_N,
                "wrong_confidence",
            )
            .copy()
        )

        false_positive[
            "error_type"
        ] = "false_positive"

        # ---------------------------------------------
        # False negatives:
        # dysarthria predicted healthy
        # ---------------------------------------------

        false_negative = (
            model_errors[
                model_errors["true_label"] == 1
            ]
            .nlargest(
                TOP_N,
                "wrong_confidence",
            )
            .copy()
        )

        false_negative[
            "error_type"
        ] = "false_negative"

        selected.extend(
            [
                false_positive,
                false_negative,
            ]
        )

    result = pd.concat(
        selected,
        ignore_index=True,
    )

    result = result.sort_values(
        [
            "model",
            "error_type",
            "wrong_confidence",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return result


# ============================================================
# HuBERT -> RESIDUAL HuBERT PREDICTION FLIPS
# ============================================================

def analyze_prediction_flips(data):

    flips = data.copy()

    # --------------------------------------------------------
    # Did the binary prediction change?
    # --------------------------------------------------------

    flips["prediction_flipped"] = (
        flips["pred_hubert"]
        != flips["pred_residual_hubert"]
    )

    # --------------------------------------------------------
    # Probability shift
    #
    # positive:
    # residualization moved prediction toward dysarthria
    #
    # negative:
    # residualization moved prediction toward healthy
    # --------------------------------------------------------

    flips["delta_p_dys"] = (
        flips["p_dys_residual_hubert"]
        - flips["p_dys_hubert"]
    )

    flips["abs_delta_p_dys"] = (
        flips["delta_p_dys"].abs()
    )

    # --------------------------------------------------------
    # Correctness before / after
    # --------------------------------------------------------

    hubert_correct = (
        flips["pred_hubert"]
        == flips["true_label"]
    )

    residual_correct = (
        flips["pred_residual_hubert"]
        == flips["true_label"]
    )

    # --------------------------------------------------------
    # Categorize every recording
    # --------------------------------------------------------

    conditions = [
        hubert_correct & residual_correct,
        hubert_correct & ~residual_correct,
        ~hubert_correct & residual_correct,
        ~hubert_correct & ~residual_correct,
    ]

    choices = [
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    ]

    flips["transition"] = np.select(
        conditions,
        choices,
        default="unknown",
    )

    # --------------------------------------------------------
    # Keep useful columns first
    # --------------------------------------------------------

    preferred_columns = [
        "dataset_index",
        "speaker_id",
        "speech_status",
        "gender",
        "fold",
        "true_label",

        "pred_hubert",
        "p_dys_hubert",

        "pred_residual_hubert",
        "p_dys_residual_hubert",

        "prediction_flipped",
        "transition",
        "delta_p_dys",
        "abs_delta_p_dys",

        "reference",
        "reference_normalized",
        "prediction",
        "wer",
        "cer",

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

        "possible_hallucination",
        "used_chunking",
        "filename",
    ]

    preferred_columns = [
        column
        for column in preferred_columns
        if column in flips.columns
    ]

    remaining_columns = [
        column
        for column in flips.columns
        if column not in preferred_columns
    ]

    flips = flips[
        preferred_columns
        + remaining_columns
    ]

    return flips


# ============================================================
# FLIP SUMMARY BY SPEAKER
# ============================================================

def summarize_flips_by_speaker(
    flips,
):

    rows = []

    for speaker_id, group in flips.groupby(
        "speaker_id"
    ):

        n = len(group)

        correct_to_wrong = (
            group["transition"]
            .eq("correct_to_wrong")
            .sum()
        )

        wrong_to_correct = (
            group["transition"]
            .eq("wrong_to_correct")
            .sum()
        )

        correct_to_correct = (
            group["transition"]
            .eq("correct_to_correct")
            .sum()
        )

        wrong_to_wrong = (
            group["transition"]
            .eq("wrong_to_wrong")
            .sum()
        )

        n_flipped = (
            group["prediction_flipped"]
            .sum()
        )

        row = {
            "speaker_id": speaker_id,
            "speech_status": (
                group["speech_status"].iloc[0]
            ),
            "fold": int(
                group["fold"].iloc[0]
            ),
            "n_recordings": n,

            "n_prediction_flips": int(
                n_flipped
            ),

            "prediction_flip_rate": (
                n_flipped / n
            ),

            "correct_to_correct": int(
                correct_to_correct
            ),

            "correct_to_wrong": int(
                correct_to_wrong
            ),

            "wrong_to_correct": int(
                wrong_to_correct
            ),

            "wrong_to_wrong": int(
                wrong_to_wrong
            ),

            "correct_to_wrong_rate": (
                correct_to_wrong / n
            ),

            "wrong_to_correct_rate": (
                wrong_to_correct / n
            ),

            "mean_delta_p_dys": (
                group["delta_p_dys"].mean()
            ),

            "mean_abs_delta_p_dys": (
                group["abs_delta_p_dys"].mean()
            ),

            "median_delta_p_dys": (
                group["delta_p_dys"].median()
            ),
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    # Speakers most harmed by residualization first
    result["net_correctness_change"] = (
        result["wrong_to_correct"]
        - result["correct_to_wrong"]
    )

    result["net_correctness_change_rate"] = (
        result["net_correctness_change"]
        / result["n_recordings"]
    )

    result = result.sort_values(
        "net_correctness_change_rate",
        ascending=True,
    )

    return result

# ============================================================
# PRINT SPEAKER SUMMARY
# ============================================================

def print_speaker_summary(
    speaker_results,
):

    print("\n" + "=" * 70)
    print("SPEAKER-LEVEL HuBERT PERFORMANCE")
    print("=" * 70)

    columns = [
        "speaker_id",
        "speech_status",
        "fold",
        "n_recordings",
        "hubert_accuracy",
        "hubert_mean_p_dys",
        "residual_hubert_accuracy",
        "residual_hubert_mean_p_dys",
    ]

    print(
        speaker_results[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    data = add_error_columns(
        data
    )

    speaker_results = (
        analyze_speakers(data)
    )

    prompt_results = (
        analyze_prompts(data)
    )

    recording_errors = (
        analyze_recording_errors(data)
    )

    top_errors = (
        get_top_confident_errors(
            recording_errors
        )
    )

    flips = analyze_prediction_flips(
        data
    )

    flip_speaker_results = (
        summarize_flips_by_speaker(
            flips
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    speaker_results.to_csv(
        SPEAKER_OUTPUT,
        index=False,
    )

    prompt_results.to_csv(
        PROMPT_OUTPUT,
        index=False,
    )

    recording_errors.to_csv(
        ERROR_OUTPUT,
        index=False,
    )

    top_errors.to_csv(
        TOP_ERRORS_OUTPUT,
        index=False,
    )
    flips.to_csv(
        FLIPS_OUTPUT,
        index=False,
    )

    flip_speaker_results.to_csv(
        FLIP_SPEAKER_OUTPUT,
        index=False,
    )
    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print_speaker_summary(
        speaker_results
    )

    print("\n" + "=" * 70)
    print("ERROR COUNTS")
    print("=" * 70)

    for model_name in MODEL_CONFIG:

        n_errors = (
            (
                data[
                    f"{model_name}_correct"
                ]
                == False
            )
            .sum()
        )

        error_rate = (
            n_errors / len(data)
        )

        print(
            f"{model_name:<20} "
            f"{n_errors:>5} errors "
            f"({error_rate:.1%})"
        )

    print("\n" + "=" * 70)
    print("HuBERT -> RESIDUAL HuBERT FLIPS BY SPEAKER")
    print("=" * 70)

    columns = [
        "speaker_id",
        "speech_status",
        "fold",
        "n_recordings",
        "prediction_flip_rate",
        "correct_to_wrong",
        "wrong_to_correct",
        "net_correctness_change_rate",
        "mean_delta_p_dys",
        "mean_abs_delta_p_dys",
    ]

    print(
        flip_speaker_results[
            columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )
    print("\nSaved:")
    print(SPEAKER_OUTPUT)
    print(PROMPT_OUTPUT)
    print(ERROR_OUTPUT)
    print(TOP_ERRORS_OUTPUT)
    print(FLIPS_OUTPUT)
    print(FLIP_SPEAKER_OUTPUT)

if __name__ == "__main__":
    main()