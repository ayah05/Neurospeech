from datasets import load_dataset


DATASET_NAME = "abnerh/TORGO-database"


def download_torgo():
    print("Loading TORGO dataset...")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset loaded successfully!")
    print(dataset)

    return dataset


if __name__ == "__main__":
    dataset = download_torgo()