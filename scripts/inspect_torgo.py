from datasets import load_dataset, Audio


DATASET_NAME = "abnerh/TORGO-database"


def main():
    print("Loading TORGO...")

    dataset = load_dataset(DATASET_NAME)
    train = dataset["train"]

    # IMPORTANT:
    # Do not decode the audio yet.
    train = train.cast_column(
        "audio",
        Audio(decode=True)
    )

    print("\nNumber of samples:")
    print(len(train))

    print("\nColumns:")
    print(train.column_names)

    print("\nFeatures:")
    print(train.features)

    print("\nFirst sample:")
    sample = train[0]

    print("Audio:", sample["audio"])
    print("Transcription:", sample["transcription"])
    print("Speech status:", sample["speech_status"])
    print("Gender:", sample["gender"])
    print("Duration:", sample["duration"])

    print("\nSpeech status values:")
    print(train.unique("speech_status"))

    print("\nGender values:")
    print(train.unique("gender"))


if __name__ == "__main__":
    main()