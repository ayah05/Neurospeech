from pathlib import Path
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from neurospeech.features.acoustic import extract_acoustic_features


DATASET_NAME = "abnerh/TORGO-database"

METADATA_PATH = ROOT / "data" / "metadata_with_folds.csv"
OUTPUT_PATH = ROOT / "data" / "acoustic_features.csv"


def get_waveform_and_sampling_rate(audio):
    """
    Convert the Hugging Face/TorchCodec audio object into
    a NumPy waveform and sampling rate.

    This helper keeps TorchCodec-specific handling outside
    the feature extraction module.
    """

    # Newer datasets versions may expose a TorchCodec decoder.
    if hasattr(audio, "get_all_samples"):

        samples = audio.get_all_samples()

        waveform = samples.data
        sampling_rate = samples.sample_rate

        # Convert torch.Tensor -> NumPy
        if hasattr(waveform, "detach"):
            waveform = (
                waveform
                .detach()
                .cpu()
                .numpy()
            )

        waveform = np.asarray(waveform)

        return waveform, sampling_rate

    # Fallback for datasets versions returning dictionary-like audio.
    if isinstance(audio, dict):

        if "array" in audio and "sampling_rate" in audio:
            return (
                np.asarray(audio["array"]),
                audio["sampling_rate"],
            )

    raise TypeError(
        f"Unsupported audio representation: {type(audio)}"
    )


def main():

    # ---------------------------------------------------------
    # LOAD METADATA
    # ---------------------------------------------------------

    print("Loading metadata...")

    metadata = pd.read_csv(METADATA_PATH)


    print(f"Metadata rows: {len(metadata)}")

    # ---------------------------------------------------------
    # LOAD TORGO
    # ---------------------------------------------------------

    print("\nLoading TORGO...")

    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]

    ###
 #   if len(train) != len(metadata):
  #      raise ValueError(
   #         "Dataset size and metadata size do not match."
   #     )
    ###
    if metadata["dataset_index"].max() >= len(train):
        raise ValueError(
            "Metadata contains an invalid dataset index."
        )

    # ---------------------------------------------------------
    # EXTRACT FEATURES
    # ---------------------------------------------------------

    print("\nExtracting acoustic features...")

    feature_rows = []

    for row in tqdm(
        metadata.itertuples(index=False),
        total=len(metadata),
        desc="Acoustic features",
    ):

        dataset_index = int(row.dataset_index)

        sample = train[dataset_index]

        try:

            waveform, sampling_rate = (
                get_waveform_and_sampling_rate(
                    sample["audio"]
                )
            )

            features = extract_acoustic_features(
                waveform=waveform,
                sampling_rate=sampling_rate,
            )

            feature_rows.append(
                {
                    "dataset_index": dataset_index,
                    "speaker_id": row.speaker_id,
                    "speech_status": row.speech_status,
                    "gender": row.gender,
                    "fold": row.fold,
                    "transcription": row.transcription,
                    "filename": row.filename,
                    **features,
                }
            )

        except Exception as error:

            print(
                f"\nWARNING: Could not process "
                f"dataset index {dataset_index}: {error}"
            )

            feature_rows.append(
                {
                    "dataset_index": dataset_index,
                    "speaker_id": row.speaker_id,
                    "speech_status": row.speech_status,
                    "gender": row.gender,
                    "fold": row.fold,
                    "transcription": row.transcription,
                    "filename": row.filename,
                    "audio_duration": np.nan,
                    "voiced_duration": np.nan,
                    "speech_ratio": np.nan,
                    "pause_ratio": np.nan,
                    "number_of_pauses": np.nan,
                    "mean_pause_duration": np.nan,
                    "f0_mean": np.nan,
                    "f0_std": np.nan,
                    "rms_mean": np.nan,
                    "rms_std": np.nan,
                }
            )

    # ---------------------------------------------------------
    # CREATE DATAFRAME
    # ---------------------------------------------------------

    features_df = pd.DataFrame(feature_rows)

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("\nValidating extracted features...")

    if len(features_df) != len(metadata):
        raise ValueError(
            "Feature table does not contain one row "
            "per metadata row."
        )

    if not features_df["dataset_index"].is_unique:
        raise ValueError(
            "Duplicate dataset indices found."
        )

    if not (
        features_df
        .groupby("speaker_id")["fold"]
        .nunique()
        .eq(1)
        .all()
    ):
        raise ValueError(
            "Speaker/fold assignments changed unexpectedly."
        )

    print("✓ One feature row per recording")
    print("✓ Dataset indices are unique")
    print("✓ Speaker-independent folds preserved")

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    feature_columns = [
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

    missing = (
        features_df[feature_columns]
        .isna()
        .sum()
    )

    print("\nMissing feature values:")
    print(missing.to_string())

    # ---------------------------------------------------------
    # FEATURE SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("ACOUSTIC FEATURE SUMMARY")
    print("=" * 80)

    print(
        features_df[feature_columns]
        .describe()
        .T
        .to_string()
    )

    # ---------------------------------------------------------
    # SUMMARY BY CLASS
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("MEAN FEATURES BY SPEECH STATUS")
    print("=" * 80)

    class_summary = (
        features_df
        .groupby("speech_status")[feature_columns]
        .mean()
    )

    print(class_summary.to_string())

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    features_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)

    print(f"\nSaved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()