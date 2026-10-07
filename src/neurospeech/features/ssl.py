import numpy as np
import torch
from transformers import AutoFeatureExtractor, HubertModel


class HubertEmbeddingExtractor:
    """
    Extract fixed-dimensional utterance embeddings from a
    pretrained HuBERT model.

    The HuBERT encoder remains frozen. Hidden representations
    are mean-pooled across time to obtain one embedding per
    utterance.
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: str | None = None,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.model_name = model_name

        print(f"Loading HuBERT: {model_name}")
        print(f"Device: {self.device}")

        self.feature_extractor = (
            AutoFeatureExtractor.from_pretrained(
                model_name
            )
        )

        self.model = HubertModel.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

        # Explicitly freeze model parameters.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.hidden_size = (
            self.model.config.hidden_size
        )

        print(
            f"HuBERT hidden size: "
            f"{self.hidden_size}"
        )

    @torch.inference_mode()
    def extract(
        self,
        waveform: np.ndarray,
        sampling_rate: int,
    ) -> np.ndarray:
        """
        Extract a mean-pooled HuBERT embedding.

        Parameters
        ----------
        waveform:
            Mono waveform as a 1D NumPy array.

        sampling_rate:
            Sampling rate of the input waveform.

        Returns
        -------
        np.ndarray
            One fixed-dimensional HuBERT embedding.
        """

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.ndim != 1:
            waveform = waveform.squeeze()

        if waveform.ndim != 1:
            raise ValueError(
                "Expected a mono 1D waveform, "
                f"got shape {waveform.shape}."
            )

        if len(waveform) == 0:
            raise ValueError(
                "Received an empty waveform."
            )

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_values = (
            inputs["input_values"]
            .to(self.device)
        )

        attention_mask = None

        if "attention_mask" in inputs:
            attention_mask = (
                inputs["attention_mask"]
                .to(self.device)
            )

        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        # Shape:
        #
        # [batch, time, hidden_size]
        #
        # Since we process one utterance here:
        #
        # [1, time, 768]
        #
        # Mean pooling over the temporal dimension gives:
        #
        # [1, 768]

        embedding = hidden_states.mean(
            dim=1
        )

        embedding = (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return embedding