# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    deskew: bool = False
    max_range: float = 100.0
    min_range: float = 3.0


class OdometryConfig(BaseModel):
    voxel_size: float = 0.5
    max_points_per_voxel: int = 20
    initial_threshold: float = 2.0
    min_motion_th: float = 0.1


class MOSConfig(BaseModel):
    voxel_size_mos: float = 0.1
    delay_mos: int = 10
    max_range_mos: float = 50.0
    min_range_mos: float = 0.0
    voxel_size_belief: float = 0.25
    max_range_belief: float = 150


class PathsConfig(BaseModel):
    """Paths as seen from inside the runtime environment."""

    data: Path = Path("data")
    cache: Optional[Path] = Path("artifacts/cache")
    logs: Path = Path("artifacts/models")


class TrainingConfig(BaseModel):
    # Reject experimental settings instead of silently running another protocol.
    model_config = ConfigDict(extra="forbid")
    id: str = "experiment_id"
    train: List[str] = Field(
        default_factory=lambda: [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "09",
            "10",
        ]
    )
    val: List[str] = Field(default_factory=lambda: ["08"])
    batch_size: int = 16
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    lr: float = 0.0001
    lr_epoch: int = 1
    lr_decay: float = 0.99
    weight_decay: float = 0.0001
    num_workers: int = 4
    dataloader: str = "cham"
    seed: int = 66
    accelerator: str = "gpu"
    devices: int = 1
    precision: str = "32-true"
    log_every_n_steps: int = 50
