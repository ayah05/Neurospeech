import re

from jiwer import wer, cer


def normalize_transcription(text):
    """
    Normalize a transcription before WER/CER evaluation.

    The same normalization MUST be applied to both
    the TORGO reference and Whisper prediction.
    """

    if text is None:
        return ""

    text = text.lower().strip()

    # Remove punctuation while preserving letters,
    # numbers and whitespace.
    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def calculate_asr_metrics(
    reference,
    prediction,
):
    """
    Calculate WER and CER after applying identical
    normalization to reference and prediction.
    """

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

    # jiwer cannot meaningfully evaluate an empty reference.
    if not reference_normalized:
        return {
            "reference_normalized":
                reference_normalized,

            "prediction_normalized":
                prediction_normalized,

            "wer": None,
            "cer": None,
        }

    return {
        "reference_normalized":
            reference_normalized,

        "prediction_normalized":
            prediction_normalized,

        "wer": wer(
            reference_normalized,
            prediction_normalized,
        ),

        "cer": cer(
            reference_normalized,
            prediction_normalized,
        ),
    }