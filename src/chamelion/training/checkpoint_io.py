"""Restricted checkpoint loading for full-state training resume."""

import torch
from pytorch_lightning.plugins.io import TorchCheckpointIO


class SafeCheckpointIO(TorchCheckpointIO):
    """Keep Lightning's save/remove behavior but restrict deserialization."""

    def load_checkpoint(self, path, map_location=None):
        # No unsafe fallback or automatic allowlisting of checkpoint objects.
        checkpoint = torch.load(
            path, map_location="cpu" if map_location is None else map_location,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a Lightning checkpoint dictionary")
        return checkpoint
