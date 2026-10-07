import os

import modal


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "facebook/hubert-base-ls960"
DATASET_NAME = "abnerh/TORGO-database"

BATCH_SIZE = 16
CHECKPOINT_EVERY = 250

# Recordings longer than this are processed in chunks instead
# of being placed into a normal padded batch.
LONG_AUDIO_THRESHOLD_SECONDS = 30.0

# Chunk size for long recordings.
CHUNK_LENGTH_SECONDS = 30.0

VOLUME_NAME = "neurospeech-hubert"
VOLUME_PATH = "/root/neurospeech-data"


# ============================================================
# MODAL SETUP
# ============================================================

app = modal.App("neurospeech-hubert")

volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "soundfile",
        "torchcodec",
    )
)


# ============================================================
# GPU WORKER
# ============================================================

@app.cls(
    image=image,
    gpu="L4",
    volumes={
        VOLUME_PATH: volume
    },
    timeout=60 * 60 * 12,
)
class HubertExtractor:

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    @modal.enter()
    def load_model(self):

        import torch

        from transformers import (
            AutoFeatureExtractor,
            HubertModel,
        )

        print(
            f"Loading HuBERT model: "
            f"{MODEL_NAME}"
        )

        self.device = torch.device(
            "cuda"
        )

        self.feature_extractor = (
            AutoFeatureExtractor.from_pretrained(
                MODEL_NAME
            )
        )

        self.model = (
            HubertModel.from_pretrained(
                MODEL_NAME
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        for parameter in (
            self.model.parameters()
        ):
            parameter.requires_grad = False

        self.hidden_size = (
            self.model.config.hidden_size
        )

        print(
            f"Model ready on "
            f"{self.device}"
        )

        print(
            f"Hidden size: "
            f"{self.hidden_size}"
        )

    # --------------------------------------------------------
    # MASK-AWARE TEMPORAL MEAN POOLING
    # --------------------------------------------------------

    def _pool_hidden_states(
        self,
        hidden_states,
        attention_mask,
    ):
        """
        Mean-pool HuBERT representations while excluding
        padded feature frames.
        """

        if attention_mask is None:
            return hidden_states.mean(
                dim=1
            )

        feature_mask = (
            self.model
            ._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
        )

        feature_mask = (
            feature_mask
            .to(hidden_states.device)
            .unsqueeze(-1)
            .to(hidden_states.dtype)
        )

        masked_hidden_states = (
            hidden_states
            * feature_mask
        )

        summed = (
            masked_hidden_states.sum(
                dim=1
            )
        )

        counts = (
            feature_mask.sum(
                dim=1
            )
            .clamp(min=1.0)
        )

        return summed / counts

    # --------------------------------------------------------
    # NORMAL BATCH EXTRACTION
    # --------------------------------------------------------

    def _extract_batch(
        self,
        waveforms,
        sampling_rate,
    ):
        """
        Extract one pooled HuBERT embedding per waveform.

        Used for normal recordings <= 30 seconds.
        """

        import numpy as np
        import torch

        inputs = (
            self.feature_extractor(
                waveforms,
                sampling_rate=sampling_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        )

        input_values = (
            inputs["input_values"]
            .to(self.device)
        )

        attention_mask = (
            inputs["attention_mask"]
            .to(self.device)
        )

        with torch.inference_mode():

            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )

            pooled = (
                self._pool_hidden_states(
                    outputs.last_hidden_state,
                    attention_mask,
                )
            )

        embeddings = (
            pooled
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # Release references to large GPU tensors.
        del inputs
        del input_values
        del attention_mask
        del outputs
        del pooled

        return embeddings

    # --------------------------------------------------------
    # LONG AUDIO EXTRACTION
    # --------------------------------------------------------

    def _extract_long_audio(
        self,
        waveform,
        sampling_rate,
    ):
        """
        Split a long recording into <=30 second chunks.

        Each chunk receives its own HuBERT embedding.

        The final recording-level embedding is a
        duration-weighted average of the chunk embeddings.
        """

        import numpy as np
        import torch

        chunk_samples = int(
            CHUNK_LENGTH_SECONDS
            * sampling_rate
        )

        chunk_embeddings = []
        chunk_lengths = []

        total_samples = len(
            waveform
        )

        for start in range(
            0,
            total_samples,
            chunk_samples,
        ):

            end = min(
                start + chunk_samples,
                total_samples,
            )

            chunk = waveform[
                start:end
            ]

            if len(chunk) == 0:
                continue

            embedding = (
                self._extract_batch(
                    waveforms=[chunk],
                    sampling_rate=sampling_rate,
                )[0]
            )

            chunk_embeddings.append(
                embedding
            )

            chunk_lengths.append(
                len(chunk)
            )

            # Helps avoid memory fragmentation after
            # processing unusually long recordings.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not chunk_embeddings:
            raise RuntimeError(
                "Long recording produced "
                "no valid chunks."
            )

        chunk_embeddings = (
            np.asarray(
                chunk_embeddings,
                dtype=np.float32,
            )
        )

        chunk_lengths = (
            np.asarray(
                chunk_lengths,
                dtype=np.float64,
            )
        )

        # Weight by actual chunk duration so that a short
        # final chunk does not count as much as a full
        # 30-second chunk.
        weights = (
            chunk_lengths
            / chunk_lengths.sum()
        )

        final_embedding = np.average(
            chunk_embeddings,
            axis=0,
            weights=weights,
        )

        return final_embedding.astype(
            np.float32
        )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    def _save_checkpoint(
        self,
        checkpoint_path,
        embeddings,
        dataset_indices,
    ):

        import numpy as np

        embeddings_array = (
            np.asarray(
                embeddings,
                dtype=np.float32,
            )
        )

        indices_array = (
            np.asarray(
                dataset_indices,
                dtype=np.int64,
            )
        )

        np.savez_compressed(
            checkpoint_path,
            embeddings=embeddings_array,
            dataset_indices=indices_array,
        )

        volume.commit()

        print(
            f"Checkpoint saved: "
            f"{len(indices_array)} samples"
        )

    # --------------------------------------------------------
    # MAIN EXTRACTION
    # --------------------------------------------------------

    @modal.method()
    def run(
        self,
        max_samples: int | None = None,
    ):

        import numpy as np
        import torch

        from datasets import (
            load_dataset,
        )

        # ====================================================
        # OUTPUT PATHS
        # ====================================================

        if max_samples is None:

            output_filename = (
                "hubert_embeddings.npz"
            )

            # IMPORTANT:
            # Keep this exact filename because our
            # previous full run already created it.
            checkpoint_filename = (
                "hubert_checkpoint_full.npz"
            )

        else:

            output_filename = (
                f"hubert_embeddings_test_"
                f"{max_samples}.npz"
            )

            checkpoint_filename = (
                f"hubert_checkpoint_test_"
                f"{max_samples}.npz"
            )

        output_path = os.path.join(
            VOLUME_PATH,
            output_filename,
        )

        checkpoint_path = os.path.join(
            VOLUME_PATH,
            checkpoint_filename,
        )

        # ====================================================
        # LOAD DATASET
        # ====================================================

        print(
            "Loading TORGO..."
        )

        dataset = load_dataset(
            DATASET_NAME,
            split="train",
        )

        dataset_size = len(
            dataset
        )

        if max_samples is None:

            total_samples = (
                dataset_size
            )

        else:

            total_samples = min(
                max_samples,
                dataset_size,
            )

        print(
            f"Dataset size: "
            f"{dataset_size}"
        )

        print(
            f"Samples to process: "
            f"{total_samples}"
        )

        # ====================================================
        # LOAD CHECKPOINT
        # ====================================================

        embeddings = []
        dataset_indices = []

        start_index = 0

        if os.path.exists(
            checkpoint_path
        ):

            print(
                "\nFound checkpoint:"
            )

            print(
                checkpoint_path
            )

            checkpoint = np.load(
                checkpoint_path
            )

            checkpoint_embeddings = (
                checkpoint[
                    "embeddings"
                ]
            )

            checkpoint_indices = (
                checkpoint[
                    "dataset_indices"
                ]
            )

            # Validate checkpoint before trusting it.
            if (
                checkpoint_embeddings.ndim
                != 2
            ):
                raise RuntimeError(
                    "Checkpoint embeddings "
                    "must be 2D."
                )

            if (
                checkpoint_embeddings.shape[1]
                != self.hidden_size
            ):
                raise RuntimeError(
                    "Checkpoint embedding "
                    "dimension does not match "
                    "HuBERT hidden size."
                )

            if (
                len(checkpoint_embeddings)
                != len(checkpoint_indices)
            ):
                raise RuntimeError(
                    "Checkpoint embeddings "
                    "and indices have "
                    "different lengths."
                )

            expected_indices = np.arange(
                len(checkpoint_indices)
            )

            if not np.array_equal(
                checkpoint_indices,
                expected_indices,
            ):
                raise RuntimeError(
                    "Checkpoint indices are "
                    "not contiguous from zero."
                )

            if not np.isfinite(
                checkpoint_embeddings
            ).all():
                raise RuntimeError(
                    "Checkpoint contains "
                    "NaN or Inf values."
                )

            embeddings = list(
                checkpoint_embeddings
            )

            dataset_indices = list(
                checkpoint_indices
            )

            start_index = len(
                dataset_indices
            )

            print(
                f"Existing embeddings: "
                f"{start_index}"
            )

            print(
                f"Resuming from dataset "
                f"index {start_index}"
            )

        else:

            print(
                "\nNo checkpoint found."
            )

            print(
                "Starting from index 0."
            )

        if start_index > total_samples:

            raise RuntimeError(
                "Checkpoint contains more "
                "samples than requested."
            )

        # ====================================================
        # PROCESS DATASET
        # ====================================================

        current_index = (
            start_index
        )

        long_audio_count = 0

        while current_index < total_samples:

            # ------------------------------------------------
            # INSPECT CURRENT SAMPLE
            # ------------------------------------------------

            sample = dataset[
                current_index
            ]

            audio = sample[
                "audio"
            ]

            waveform = np.asarray(
                audio["array"],
                dtype=np.float32,
            )

            if waveform.ndim != 1:
                waveform = (
                    waveform.squeeze()
                )

            if waveform.ndim != 1:
                raise RuntimeError(
                    f"Dataset index "
                    f"{current_index}: "
                    f"expected mono audio, "
                    f"got {waveform.shape}"
                )

            sampling_rate = int(
                audio[
                    "sampling_rate"
                ]
            )

            duration_seconds = (
                len(waveform)
                / sampling_rate
            )

            # ------------------------------------------------
            # LONG AUDIO
            # ------------------------------------------------

            if (
                duration_seconds
                > LONG_AUDIO_THRESHOLD_SECONDS
            ):

                print(
                    f"\nLong recording at "
                    f"index {current_index}: "
                    f"{duration_seconds:.2f}s"
                )

                print(
                    "Processing with "
                    "30-second chunks..."
                )

                embedding = (
                    self._extract_long_audio(
                        waveform=waveform,
                        sampling_rate=sampling_rate,
                    )
                )

                if embedding.shape != (
                    self.hidden_size,
                ):

                    raise RuntimeError(
                        "Unexpected long-audio "
                        "embedding shape: "
                        f"{embedding.shape}"
                    )

                if not np.isfinite(
                    embedding
                ).all():

                    raise RuntimeError(
                        f"NaN/Inf in embedding "
                        f"at index "
                        f"{current_index}"
                    )

                embeddings.append(
                    embedding
                )

                dataset_indices.append(
                    current_index
                )

                long_audio_count += 1

                current_index += 1

            # ------------------------------------------------
            # NORMAL BATCH
            # ------------------------------------------------

            else:

                batch_waveforms = []
                batch_indices = []

                batch_sampling_rate = (
                    sampling_rate
                )

                # Build a batch of consecutive normal
                # recordings. Stop before a long one.
                while (
                    current_index
                    < total_samples
                    and len(
                        batch_waveforms
                    )
                    < BATCH_SIZE
                ):

                    batch_sample = (
                        dataset[
                            current_index
                        ]
                    )

                    batch_audio = (
                        batch_sample[
                            "audio"
                        ]
                    )

                    batch_waveform = (
                        np.asarray(
                            batch_audio[
                                "array"
                            ],
                            dtype=np.float32,
                        )
                    )

                    if (
                        batch_waveform.ndim
                        != 1
                    ):
                        batch_waveform = (
                            batch_waveform
                            .squeeze()
                        )

                    if (
                        batch_waveform.ndim
                        != 1
                    ):
                        raise RuntimeError(
                            f"Dataset index "
                            f"{current_index}: "
                            f"expected mono audio."
                        )

                    batch_sr = int(
                        batch_audio[
                            "sampling_rate"
                        ]
                    )

                    batch_duration = (
                        len(
                            batch_waveform
                        )
                        / batch_sr
                    )

                    # Do not add a long recording
                    # to the padded batch.
                    if (
                        batch_duration
                        > LONG_AUDIO_THRESHOLD_SECONDS
                    ):
                        break

                    if (
                        batch_sr
                        != batch_sampling_rate
                    ):
                        raise RuntimeError(
                            "Sampling rate changed "
                            "inside batch."
                        )

                    batch_waveforms.append(
                        batch_waveform
                    )

                    batch_indices.append(
                        current_index
                    )

                    current_index += 1

                # This should only be empty when the
                # next sample is long. Let the outer
                # loop handle it.
                if not batch_waveforms:
                    continue

                batch_embeddings = (
                    self._extract_batch(
                        waveforms=batch_waveforms,
                        sampling_rate=(
                            batch_sampling_rate
                        ),
                    )
                )

                expected_shape = (
                    len(batch_indices),
                    self.hidden_size,
                )

                if (
                    batch_embeddings.shape
                    != expected_shape
                ):

                    raise RuntimeError(
                        "Unexpected batch "
                        "embedding shape: "
                        f"{batch_embeddings.shape}"
                    )

                if not np.isfinite(
                    batch_embeddings
                ).all():

                    raise RuntimeError(
                        "NaN or Inf detected "
                        "in batch."
                    )

                embeddings.extend(
                    batch_embeddings
                )

                dataset_indices.extend(
                    batch_indices
                )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            processed = len(
                dataset_indices
            )

            print(
                f"Processed "
                f"{processed}/"
                f"{total_samples}"
            )

            # ------------------------------------------------
            # CHECKPOINT
            # ------------------------------------------------

            # Save whenever we have crossed another
            # CHECKPOINT_EVERY boundary.
            #
            # Example:
            # 13264 -> 13520 crosses 13500.
            previous_processed = (
                processed - 1
            )

            crossed_checkpoint = (
                processed
                // CHECKPOINT_EVERY
                >
                previous_processed
                // CHECKPOINT_EVERY
            )

            if (
                crossed_checkpoint
                or processed
                == total_samples
            ):

                print(
                    "\nSaving checkpoint..."
                )

                self._save_checkpoint(
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    embeddings=embeddings,
                    dataset_indices=(
                        dataset_indices
                    ),
                )

            # Release cached unused CUDA blocks.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ====================================================
        # FINAL ARRAYS
        # ====================================================

        embeddings_array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        dataset_indices_array = (
            np.asarray(
                dataset_indices,
                dtype=np.int64,
            )
        )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        print(
            "\nFinal validation:"
        )

        print(
            f"Embeddings shape: "
            f"{embeddings_array.shape}"
        )

        print(
            f"Dataset indices shape: "
            f"{dataset_indices_array.shape}"
        )

        print(
            f"Long recordings "
            f"processed in this run: "
            f"{long_audio_count}"
        )

        expected_shape = (
            total_samples,
            self.hidden_size,
        )

        if (
            embeddings_array.shape
            != expected_shape
        ):

            raise RuntimeError(
                "Final embedding matrix "
                "has unexpected shape. "
                f"Expected "
                f"{expected_shape}, "
                f"got "
                f"{embeddings_array.shape}."
            )

        if (
            dataset_indices_array.shape
            != (total_samples,)
        ):

            raise RuntimeError(
                "Final dataset index array "
                "has unexpected shape."
            )

        expected_indices = np.arange(
            total_samples,
            dtype=np.int64,
        )

        if not np.array_equal(
            dataset_indices_array,
            expected_indices,
        ):

            raise RuntimeError(
                "Dataset indices are not "
                "complete and ordered."
            )

        if not np.isfinite(
            embeddings_array
        ).all():

            raise RuntimeError(
                "Final embeddings contain "
                "NaN or Inf values."
            )

        # ====================================================
        # SAVE FINAL OUTPUT
        # ====================================================

        np.savez_compressed(
            output_path,
            embeddings=embeddings_array,
            dataset_indices=(
                dataset_indices_array
            ),
        )

        volume.commit()

        print(
            "\nSaved final embeddings to:"
        )

        print(
            output_path
        )

        print(
            "\nHuBERT extraction complete."
        )

        return {
            "samples": int(
                total_samples
            ),
            "embedding_dim": int(
                embeddings_array.shape[1]
            ),
            "long_recordings_this_run": int(
                long_audio_count
            ),
            "output_path": output_path,
        }


# ============================================================
# LOCAL ENTRYPOINT
# ============================================================

@app.local_entrypoint()
def main(
    full: bool = False,
):

    extractor = (
        HubertExtractor()
    )

    if full:

        max_samples = None

        print(
            "Starting FULL "
            "HuBERT extraction..."
        )

    else:

        max_samples = 64

        print(
            "Starting HuBERT test "
            "extraction for "
            "64 samples..."
        )

    result = (
        extractor.run.remote(
            max_samples=max_samples
        )
    )

    print(
        "\nExtraction result:"
    )

    print(
        result
    )