"""Shared inference-model loading, independent of evaluation experiments."""
from pathlib import Path

import torch

from chamelion.models.network import ChamelionNet


def load_model(checkpoint: Path, voxel_size: float, device: torch.device) -> ChamelionNet:
    """Load a trusted checkpoint without changing its weights or model settings.

    Accept a Lightning state_dict wrapper or a bare state dict, optionally using
    the training module's ``mos.`` prefix. Only load checkpoints you trust:
    full Lightning checkpoints require pickle deserialization.
    """
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = loaded.get("state_dict", loaded)
    state_dict = {
        (key.removeprefix("mos.") if key.startswith("mos.") else key): value
        for key, value in state_dict.items()
    }
    model = ChamelionNet(voxel_size)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    model.freeze()
    return model
