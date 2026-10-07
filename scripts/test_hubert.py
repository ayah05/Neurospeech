from datasets import load_dataset

from src.neurospeech.features.ssl import (
    HubertEmbeddingExtractor,
)


# ============================================================
# LOAD ONE TORGO SAMPLE
# ============================================================

print("Loading TORGO...")

dataset = load_dataset(
    "abnerh/TORGO-database",
    split="train",
)


sample = dataset[0]

audio = sample["audio"]


print("\nSample information:")
print(f"Transcription: {sample['transcription']}")
print(f"Speech status: {sample['speech_status']}")
print(f"Sampling rate: {audio['sampling_rate']}")
print(f"Samples: {len(audio['array'])}")


# ============================================================
# LOAD HUBERT
# ============================================================

extractor = HubertEmbeddingExtractor()


# ============================================================
# EXTRACT EMBEDDING
# ============================================================

embedding = extractor.extract(
    waveform=audio["array"],
    sampling_rate=audio["sampling_rate"],
)


# ============================================================
# VALIDATE
# ============================================================

print("\nEmbedding:")
print(f"Shape: {embedding.shape}")
print(f"dtype: {embedding.dtype}")
print(f"Mean: {embedding.mean():.6f}")
print(f"Std: {embedding.std():.6f}")
print(f"Min: {embedding.min():.6f}")
print(f"Max: {embedding.max():.6f}")


if embedding.shape != (
    extractor.hidden_size,
):
    raise RuntimeError(
        "Unexpected embedding shape: "
        f"{embedding.shape}"
    )


print(
    "\nHuBERT smoke test successful."
)