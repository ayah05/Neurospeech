from pathlib import Path
import shutil

import pandas as pd
import soundfile as sf
from datasets import load_dataset


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FLIPS_PATH = (
    ROOT
    / "results"
    / "prediction_flips_hubert_residual.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "audio_error_cases"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "audio_error_cases.csv"
)


# ============================================================
# CONFIG
# ============================================================

CASES = {
    "F04": "correct_to_wrong",
    "F03": "wrong_to_correct",
    "FC01": "correct_to_wrong",
    "FC03": "wrong_to_correct",
}

N_PER_SPEAKER = 10


# ============================================================
# SELECT CASES
# ============================================================

def select_cases(flips):

    selected = []

    for speaker_id, transition in CASES.items():

        subset = flips[
            (flips["speaker_id"] == speaker_id)
            & (flips["transition"] == transition)
        ].copy()

        print(
            f"{speaker_id}: "
            f"{len(subset)} available "
            f"{transition} cases"
        )

        # Strongest probability changes first
        subset = subset.sort_values(
            "abs_delta_p_dys",
            ascending=False,
        )

        subset = subset.head(
            N_PER_SPEAKER
        )

        selected.append(subset)

    result = pd.concat(
        selected,
        ignore_index=True,
    )

    return result


# ============================================================
# EXPORT AUDIO
# ============================================================

def export_audio(selected):

    print("\nLoading TORGO...")

    dataset = load_dataset(
        "abnerh/TORGO-database",
        split="train",
    )

    exported_rows = []

    for _, row in selected.iterrows():

        dataset_index = int(
            row["dataset_index"]
        )

        speaker_id = row[
            "speaker_id"
        ]

        transition = row[
            "transition"
        ]

        sample = dataset[
            dataset_index
        ]

        audio = sample[
            "audio"
        ]

        # Hugging Face Audio object
        array = audio[
            "array"
        ]

        sampling_rate = audio[
            "sampling_rate"
        ]

        speaker_dir = (
            OUTPUT_DIR
            / speaker_id
            / transition
        )

        speaker_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prompt = str(
            row.get(
                "reference_normalized",
                "unknown",
            )
        )

        # Keep filename safe
        safe_prompt = (
            prompt
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        safe_prompt = safe_prompt[:40]

        filename = (
            f"{dataset_index}_"
            f"{safe_prompt}.wav"
        )

        output_path = (
            speaker_dir
            / filename
        )

        sf.write(
            output_path,
            array,
            sampling_rate,
        )

        result_row = row.to_dict()

        result_row[
            "exported_audio"
        ] = str(
            output_path.relative_to(
                ROOT
            )
        )

        exported_rows.append(
            result_row
        )

        print(
            f"Exported: "
            f"{speaker_id:<4} | "
            f"{transition:<16} | "
            f"{prompt:<20} | "
            f"ΔP={row['delta_p_dys']:+.3f}"
        )

    return pd.DataFrame(
        exported_rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    flips = pd.read_csv(
        FLIPS_PATH
    )

    print("=" * 70)
    print("Selecting audio error cases")
    print("=" * 70)

    selected = select_cases(
        flips
    )

    print(
        f"\nSelected recordings: "
        f"{len(selected)}"
    )

    exported = export_audio(
        selected
    )

    exported.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Exported {len(exported)} recordings."
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    print(
        f"Audio directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()