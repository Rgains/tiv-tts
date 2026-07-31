"""A very small diagnostic text-to-mel model for pipeline smoke testing."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TinyTTS(nn.Module):
    """Uniformly align encoded characters to mel frames and predict log-mels.

    This model is deliberately small and is not intended for production speech.
    Its purpose is to exercise tokenization, batching, gradients, checkpoints,
    resume behavior, and waveform generation before cloud training.
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        n_mels: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, n_mels, kernel_size=1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        text_lengths: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Predict padded log-mels shaped ``[batch, n_mels, max_frames]``."""

        encoded = self.encoder(self.embedding(tokens).transpose(1, 2))
        max_frames = int(output_lengths.max().item())
        stretched: list[torch.Tensor] = []
        for item in range(tokens.shape[0]):
            text_length = int(text_lengths[item].item())
            output_length = int(output_lengths[item].item())
            sequence = encoded[item : item + 1, :, :text_length]
            aligned = F.interpolate(
                sequence,
                size=output_length,
                mode="linear",
                align_corners=False,
            )
            stretched.append(F.pad(aligned, (0, max_frames - output_length)))
        return self.decoder(torch.cat(stretched, dim=0))

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, output_frames: int) -> torch.Tensor:
        """Predict a single log-mel from one token sequence."""

        text_lengths = torch.tensor([tokens.shape[1]], device=tokens.device)
        output_lengths = torch.tensor([output_frames], device=tokens.device)
        return self(tokens, text_lengths, output_lengths)[0]

