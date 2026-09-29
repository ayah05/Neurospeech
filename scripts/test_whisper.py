from pathlib import Path
import sys

import numpy as np
import torch

from datasets import load_dataset
from transformers import pipeline


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )


from neurospeech.evaluation.asr import (
    calculate_asr_metrics,
)


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "abnerh/TORGO-database"

MODEL_NAME = "openai/whisper-small"


# ============================================================
# AUDIO HELPER
# ============================================================

def get_waveform_and_sampling_rate(audio):

    if hasattr(
        audio,
        "get_all_samples"
    ):

        samples = (
            audio.get_all_samples()
        )

        waveform = samples.data
        sampling_rate = (
            samples.sample_rate
        )

        if hasattr(
            waveform,
            "detach"
        ):
            waveform = (
                waveform
                .detach()
                .cpu()
                .numpy()
            )

        waveform = np.asarray(
            waveform
        )

        # Convert [channels, samples]
        # to mono.
        if waveform.ndim > 1:
            waveform = waveform.mean(
                axis=0
            )

        return (
            waveform,
            sampling_rate
        )

    raise TypeError(
        "Unsupported audio representation: "
        f"{type(audio)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading TORGO..."
    )

    dataset = load_dataset(
        DATASET_NAME
    )

    train = dataset["train"]


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = 0

        print(
            "Using GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        device = -1

        print(
            "Using CPU"
        )


    # --------------------------------------------------------
    # WHISPER
    # --------------------------------------------------------

    print(
        "\nLoading Whisper..."
    )

    asr = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
    )


    # --------------------------------------------------------
    # TEST SAMPLES
    # --------------------------------------------------------

    #
    # From our previous audit:
    #
    # sample 0     = healthy
    # sample 10978 = dysarthria
    #

    test_indices = [
        0,
        10978,
    ]


    for index in test_indices:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"SAMPLE {index}"
        )

        print(
            "=" * 70
        )

        sample = train[index]

        waveform, sampling_rate = (
            get_waveform_and_sampling_rate(
                sample["audio"]
            )
        )


        # ----------------------------------------------------
        # WHISPER INFERENCE
        # ----------------------------------------------------

        result = asr(
            {
                "array": waveform,
                "sampling_rate":
                    sampling_rate,
            },
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
            },
        )

        prediction = (
            result["text"]
        )

        reference = (
            sample["transcription"]
        )


        # ----------------------------------------------------
        # EVALUATION
        # ----------------------------------------------------

        metrics = (
            calculate_asr_metrics(
                reference=reference,
                prediction=prediction,
            )
        )


        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Status: "
            f"{sample['speech_status']}"
        )

        print(
            f"\nReference:\n"
            f"{reference}"
        )

        print(
            f"\nWhisper:\n"
            f"{prediction}"
        )

        print(
            "\nNormalized reference:"
        )

        print(
            metrics[
                "reference_normalized"
            ]
        )

        print(
            "\nNormalized prediction:"
        )

        print(
            metrics[
                "prediction_normalized"
            ]
        )

        print(
            f"\nWER: "
            f"{metrics['wer']}"
        )

        print(
            f"CER: "
            f"{metrics['cer']}"
        )


if __name__ == "__main__":
    main()