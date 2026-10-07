from pathlib import Path
import io

import modal


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "neurospeech-whisper-full"

DATASET_NAME = "abnerh/TORGO-database"
MODEL_NAME = "openai/whisper-small"

BATCH_SIZE = 16

# Whisper normally operates on ~30-second windows.
LONG_AUDIO_THRESHOLD_SECONDS = 30.0
CHUNK_LENGTH_SECONDS = 30

# Persist progress regularly.
CHECKPOINT_EVERY = 250

# Diagnostic heuristic only.
# Nothing is removed based on this threshold.
HALLUCINATION_LENGTH_RATIO = 5.0


# ============================================================
# LOCAL PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = (
    ROOT / "data" / "metadata_with_folds.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "whisper_full_predictions.csv"
)


# ============================================================
# MODAL
# ============================================================

app = modal.App(APP_NAME)


# Persistent storage.
volume = modal.Volume.from_name(
    "neurospeech-whisper",
    create_if_missing=True,
)

VOLUME_PATH = "/neurospeech"

CHECKPOINT_PATH = (
    f"{VOLUME_PATH}/whisper_checkpoint.csv"
)

FINAL_REMOTE_PATH = (
    f"{VOLUME_PATH}/whisper_full_predictions.csv"
)


# ============================================================
# CONTAINER IMAGE
# ============================================================

image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
    .apt_install(
        "ffmpeg"
    )
    .pip_install(
        "torch",
        "torchaudio",
        "torchcodec",
        "transformers",
        "datasets",
        "accelerate",
        "jiwer",
        "pandas",
        "numpy",
        "tqdm",
        "huggingface-hub",
    )
)


# ============================================================
# REMOTE GPU FUNCTION
# ============================================================

@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 6,
    volumes={
        VOLUME_PATH: volume
    },
)
def run_whisper(metadata_csv):

    import os
    import re

    import numpy as np
    import pandas as pd
    import torch

    from datasets import load_dataset
    from jiwer import wer, cer
    from tqdm import tqdm
    from transformers import pipeline


    # ========================================================
    # HELPERS
    # ========================================================

    def normalize_transcription(text):

        if text is None:
            return ""

        text = str(text).lower().strip()

        text = re.sub(
            r"[^\w\s]",
            "",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()


    def calculate_metrics(
        reference,
        prediction,
    ):

        reference_normalized = (
            normalize_transcription(
                reference
            )
        )

        prediction_normalized = (
            normalize_transcription(
                prediction
            )
        )

        reference_words = (
            reference_normalized.split()
        )

        prediction_words = (
            prediction_normalized.split()
        )

        reference_word_count = len(
            reference_words
        )

        prediction_word_count = len(
            prediction_words
        )

        reference_character_count = len(
            reference_normalized
        )

        prediction_character_count = len(
            prediction_normalized
        )


        # ----------------------------------------------------
        # WER / CER
        # ----------------------------------------------------

        if reference_normalized:

            sample_wer = wer(
                reference_normalized,
                prediction_normalized,
            )

            sample_cer = cer(
                reference_normalized,
                prediction_normalized,
            )

        else:

            sample_wer = np.nan
            sample_cer = np.nan


        # ----------------------------------------------------
        # LENGTH RATIO
        # ----------------------------------------------------

        if reference_word_count > 0:

            length_ratio = (
                prediction_word_count
                / reference_word_count
            )

        else:

            length_ratio = np.nan


        # ----------------------------------------------------
        # DIAGNOSTIC FLAG
        # ----------------------------------------------------

        possible_hallucination = bool(
            reference_word_count > 0
            and length_ratio
            > HALLUCINATION_LENGTH_RATIO
        )


        return {
            "reference_normalized":
                reference_normalized,

            "prediction_normalized":
                prediction_normalized,

            "wer":
                sample_wer,

            "cer":
                sample_cer,

            "reference_word_count":
                reference_word_count,

            "prediction_word_count":
                prediction_word_count,

            "reference_character_count":
                reference_character_count,

            "prediction_character_count":
                prediction_character_count,

            "length_ratio":
                length_ratio,

            "possible_hallucination":
                possible_hallucination,
        }


    def get_audio(audio):

        if not hasattr(
            audio,
            "get_all_samples",
        ):

            raise TypeError(
                "Unsupported audio type: "
                f"{type(audio)}"
            )


        samples = (
            audio.get_all_samples()
        )

        waveform = samples.data

        sampling_rate = (
            samples.sample_rate
        )


        if hasattr(
            waveform,
            "detach",
        ):

            waveform = (
                waveform
                .detach()
                .cpu()
                .numpy()
            )


        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )


        # Multi-channel -> mono
        if waveform.ndim > 1:

            waveform = waveform.mean(
                axis=0
            )


        duration = (
            len(waveform)
            / sampling_rate
        )


        return {
            "array": waveform,
            "sampling_rate": sampling_rate,
        }, duration


    def create_error_result(
        row,
        error,
        audio_duration=np.nan,
    ):

        return {
            "dataset_index":
                int(row["dataset_index"]),

            "speaker_id":
                row["speaker_id"],

            "speech_status":
                row["speech_status"],

            "gender":
                row["gender"],

            "fold":
                row["fold"],

            "filename":
                row["filename"],

            "reference":
                row["transcription"],

            "prediction":
                None,

            "reference_normalized":
                None,

            "prediction_normalized":
                None,

            "wer":
                np.nan,

            "cer":
                np.nan,

            "reference_word_count":
                np.nan,

            "prediction_word_count":
                np.nan,

            "reference_character_count":
                np.nan,

            "prediction_character_count":
                np.nan,

            "length_ratio":
                np.nan,

            "possible_hallucination":
                False,

            "audio_duration":
                audio_duration,

            "used_chunking":
                False,

            "error":
                str(error),
        }


    def create_success_result(
        row,
        prediction,
        audio_duration,
        used_chunking,
    ):

        metrics = calculate_metrics(
            reference=row[
                "transcription"
            ],
            prediction=prediction,
        )


        return {
            "dataset_index":
                int(row["dataset_index"]),

            "speaker_id":
                row["speaker_id"],

            "speech_status":
                row["speech_status"],

            "gender":
                row["gender"],

            "fold":
                row["fold"],

            "filename":
                row["filename"],

            "reference":
                row["transcription"],

            "prediction":
                prediction,

            **metrics,

            "audio_duration":
                audio_duration,

            "used_chunking":
                used_chunking,

            "error":
                None,
        }


    def save_checkpoint(
        results,
    ):

        if not results:
            return


        checkpoint_df = pd.DataFrame(
            results
        )


        checkpoint_df = (
            checkpoint_df
            .drop_duplicates(
                subset=[
                    "dataset_index"
                ],
                keep="last",
            )
            .sort_values(
                "dataset_index"
            )
            .reset_index(
                drop=True
            )
        )


        checkpoint_df.to_csv(
            CHECKPOINT_PATH,
            index=False,
        )


        # Persist the file in Modal Volume.
        volume.commit()


    # ========================================================
    # LOAD METADATA
    # ========================================================

    print(
        "Loading metadata..."
    )


    metadata = pd.read_csv(
        io.BytesIO(
            metadata_csv
        )
    )


    required_columns = {
        "dataset_index",
        "speaker_id",
        "speech_status",
        "gender",
        "fold",
        "filename",
        "transcription",
    }


    missing_columns = (
        required_columns
        - set(metadata.columns)
    )


    if missing_columns:

        raise ValueError(
            "Metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )


    if not metadata[
        "dataset_index"
    ].is_unique:

        raise ValueError(
            "dataset_index must be unique."
        )


    total = len(metadata)


    print(
        f"Recordings: {total}"
    )

    print(
        "Speakers: "
        f"{metadata['speaker_id'].nunique()}"
    )


    # ========================================================
    # LOAD EXISTING CHECKPOINT
    # ========================================================

    all_results = []


    if os.path.exists(
        CHECKPOINT_PATH
    ):

        print(
            "Existing checkpoint found."
        )


        checkpoint_df = pd.read_csv(
            CHECKPOINT_PATH
        )


        checkpoint_df = (
            checkpoint_df
            .drop_duplicates(
                subset=[
                    "dataset_index"
                ],
                keep="last",
            )
        )


        all_results = (
            checkpoint_df
            .to_dict(
                orient="records"
            )
        )


        print(
            "Checkpoint contains "
            f"{len(all_results)} recordings."
        )


    completed_indices = {
        int(result["dataset_index"])
        for result in all_results
    }


    remaining_metadata = (
        metadata[
            ~metadata[
                "dataset_index"
            ].isin(
                completed_indices
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    remaining = len(
        remaining_metadata
    )


    print(
        f"Remaining recordings: "
        f"{remaining}"
    )


    # ========================================================
    # NOTHING LEFT?
    # ========================================================

    if remaining == 0:

        print(
            "All recordings already exist "
            "in checkpoint."
        )

        final_df = pd.DataFrame(
            all_results
        )

        final_df = (
            final_df
            .sort_values(
                "dataset_index"
            )
            .reset_index(
                drop=True
            )
        )

        buffer = io.StringIO()

        final_df.to_csv(
            buffer,
            index=False,
        )

        return buffer.getvalue()


    # ========================================================
    # LOAD TORGO
    # ========================================================

    print(
        "Loading TORGO..."
    )


    dataset = load_dataset(
        DATASET_NAME
    )

    train = dataset["train"]


    if len(train) != total:

        raise ValueError(
            "Dataset/metadata size mismatch: "
            f"TORGO={len(train)}, "
            f"metadata={total}"
        )


    # ========================================================
    # GPU
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )


    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


    # ========================================================
    # WHISPER
    # ========================================================

    print(
        f"Loading {MODEL_NAME}..."
    )


    asr = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=0,
        dtype=torch.float16,
    )


    print(
        "Whisper loaded."
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    processed_since_checkpoint = 0


    progress_bar = tqdm(
        total=remaining,
        desc="Whisper",
        unit="recording",
        mininterval=5.0,
        dynamic_ncols=False,
    )


    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_start in range(
        0,
        remaining,
        BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + BATCH_SIZE,
            remaining,
        )


        batch_df = (
            remaining_metadata.iloc[
                batch_start:batch_end
            ]
        )


        # ----------------------------------------------------
        # Separate normal and long recordings.
        # ----------------------------------------------------

        normal_audio = []
        normal_rows = []
        normal_durations = []

        long_audio = []
        long_rows = []
        long_durations = []


        # ====================================================
        # LOAD AUDIO
        # ====================================================

        for _, row in (
            batch_df.iterrows()
        ):

            dataset_index = int(
                row["dataset_index"]
            )


            try:

                sample = train[
                    dataset_index
                ]


                (
                    audio_input,
                    audio_duration,
                ) = get_audio(
                    sample["audio"]
                )


                if (
                    audio_duration
                    > LONG_AUDIO_THRESHOLD_SECONDS
                ):

                    long_audio.append(
                        audio_input
                    )

                    long_rows.append(
                        row
                    )

                    long_durations.append(
                        audio_duration
                    )

                else:

                    normal_audio.append(
                        audio_input
                    )

                    normal_rows.append(
                        row
                    )

                    normal_durations.append(
                        audio_duration
                    )


            except Exception as error:

                all_results.append(
                    create_error_result(
                        row=row,
                        error=error,
                    )
                )

                processed_since_checkpoint += 1
                progress_bar.update(1)


        # ====================================================
        # NORMAL BATCHED AUDIO
        # ====================================================

        if normal_audio:

            try:

                predictions = asr(
                    normal_audio,
                    batch_size=min(
                        BATCH_SIZE,
                        len(normal_audio),
                    ),
                    generate_kwargs={
                        "language":
                            "english",

                        "task":
                            "transcribe",
                    },
                )


                if isinstance(
                    predictions,
                    dict,
                ):

                    predictions = [
                        predictions
                    ]


                if (
                    len(predictions)
                    != len(normal_rows)
                ):

                    raise RuntimeError(
                        "Unexpected prediction count: "
                        f"{len(predictions)} for "
                        f"{len(normal_rows)} inputs."
                    )


                for (
                    row,
                    duration,
                    prediction_result,
                ) in zip(
                    normal_rows,
                    normal_durations,
                    predictions,
                ):

                    prediction = (
                        prediction_result.get(
                            "text"
                        )
                    )


                    if prediction is None:

                        all_results.append(
                            create_error_result(
                                row=row,
                                error=(
                                    "Whisper returned "
                                    "no text."
                                ),
                                audio_duration=duration,
                            )
                        )

                    else:

                        all_results.append(
                            create_success_result(
                                row=row,
                                prediction=prediction,
                                audio_duration=duration,
                                used_chunking=False,
                            )
                        )


                    processed_since_checkpoint += 1
                    progress_bar.update(1)


            # ------------------------------------------------
            # If the batch itself fails, retry each recording.
            # ------------------------------------------------

            except Exception as batch_error:

                tqdm.write(
                    "Normal batch failed; "
                    "retrying individually. "
                    f"Reason: {batch_error}"
                )


                for (
                    row,
                    audio_input,
                    duration,
                ) in zip(
                    normal_rows,
                    normal_audio,
                    normal_durations,
                ):

                    try:

                        prediction_result = asr(
                            audio_input,
                            generate_kwargs={
                                "language":
                                    "english",

                                "task":
                                    "transcribe",
                            },
                        )


                        prediction = (
                            prediction_result.get(
                                "text"
                            )
                        )


                        if prediction is None:

                            raise RuntimeError(
                                "Whisper returned "
                                "no text."
                            )


                        all_results.append(
                            create_success_result(
                                row=row,
                                prediction=prediction,
                                audio_duration=duration,
                                used_chunking=False,
                            )
                        )


                    except Exception as error:

                        all_results.append(
                            create_error_result(
                                row=row,
                                error=error,
                                audio_duration=duration,
                            )
                        )


                    processed_since_checkpoint += 1
                    progress_bar.update(1)


        # ====================================================
        # LONG AUDIO
        # ====================================================
        #
        # Process these separately using 30-second chunking.
        #
        # Hugging Face's ASR pipeline supports chunk_length_s
        # specifically for long audio.
        # ====================================================

        for (
            row,
            audio_input,
            duration,
        ) in zip(
            long_rows,
            long_audio,
            long_durations,
        ):

            try:

                prediction_result = asr(
                    audio_input,
                    chunk_length_s=(
                        CHUNK_LENGTH_SECONDS
                    ),
                    batch_size=8,
                    generate_kwargs={
                        "language":
                            "english",

                        "task":
                            "transcribe",
                    },
                )


                prediction = (
                    prediction_result.get(
                        "text"
                    )
                )


                if prediction is None:

                    raise RuntimeError(
                        "Whisper returned no text "
                        "for long recording."
                    )


                all_results.append(
                    create_success_result(
                        row=row,
                        prediction=prediction,
                        audio_duration=duration,
                        used_chunking=True,
                    )
                )


            except Exception as error:

                all_results.append(
                    create_error_result(
                        row=row,
                        error=error,
                        audio_duration=duration,
                    )
                )


            processed_since_checkpoint += 1
            progress_bar.update(1)


        # ====================================================
        # CHECKPOINT
        # ====================================================

        if (
            processed_since_checkpoint
            >= CHECKPOINT_EVERY
        ):

            save_checkpoint(
                all_results
            )


            tqdm.write(
                "Checkpoint saved: "
                f"{len(all_results)}/{total}"
            )


            processed_since_checkpoint = 0


    # ========================================================
    # CLOSE PROGRESS BAR
    # ========================================================

    progress_bar.close()


    # ========================================================
    # SAVE LAST CHECKPOINT
    # ========================================================

    save_checkpoint(
        all_results
    )


    # ========================================================
    # FINAL DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )


    results_df = (
        results_df
        .drop_duplicates(
            subset=[
                "dataset_index"
            ],
            keep="last",
        )
        .sort_values(
            "dataset_index"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print(
        "Validating final results..."
    )


    if len(results_df) != total:

        raise ValueError(
            "Result count mismatch: "
            f"results={len(results_df)}, "
            f"expected={total}"
        )


    if not results_df[
        "dataset_index"
    ].is_unique:

        raise ValueError(
            "Duplicate dataset indices found."
        )


    expected_indices = set(
        metadata[
            "dataset_index"
        ].astype(int)
    )


    result_indices = set(
        results_df[
            "dataset_index"
        ].astype(int)
    )


    if (
        result_indices
        != expected_indices
    ):

        missing = (
            expected_indices
            - result_indices
        )

        extra = (
            result_indices
            - expected_indices
        )


        raise ValueError(
            "Dataset index mismatch. "
            f"Missing={sorted(missing)[:10]}, "
            f"extra={sorted(extra)[:10]}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    successful = int(
        results_df[
            "error"
        ].isna().sum()
    )


    failed = int(
        results_df[
            "error"
        ].notna().sum()
    )


    hallucinations = int(
        results_df[
            "possible_hallucination"
        ]
        .fillna(False)
        .sum()
    )


    chunked = int(
        results_df[
            "used_chunking"
        ]
        .fillna(False)
        .sum()
    )


    print(
        "\nInference complete."
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        "Possible hallucinations: "
        f"{hallucinations}"
    )

    print(
        "Long recordings processed "
        f"with chunking: {chunked}"
    )


    # ========================================================
    # SAVE FINAL RESULT TO VOLUME
    # ========================================================

    results_df.to_csv(
        FINAL_REMOTE_PATH,
        index=False,
    )


    volume.commit()


    # ========================================================
    # RETURN CSV TO LOCAL MACHINE
    # ========================================================

    buffer = io.StringIO()


    results_df.to_csv(
        buffer,
        index=False,
    )


    return buffer.getvalue()


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main():

    print(
        "Reading local metadata..."
    )


    if not METADATA_PATH.exists():

        raise FileNotFoundError(
            "Metadata not found: "
            f"{METADATA_PATH}"
        )


    metadata_bytes = (
        METADATA_PATH.read_bytes()
    )


    print(
        "Starting/resuming full "
        "Whisper inference on Modal..."
    )


    result_csv = (
        run_whisper.remote(
            metadata_bytes
        )
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    OUTPUT_PATH.write_text(
        result_csv,
        encoding="utf-8",
    )


    print(
        "\nFull Whisper predictions "
        "saved locally to:"
    )

    print(
        OUTPUT_PATH
    )