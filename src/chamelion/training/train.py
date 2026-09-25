#!/usr/bin/env python3
# Chamelion modifications are licensed under GPL-3.0-or-later.
# The original MapMOS copyright and MIT license notice follow.

# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
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
from typing import Optional

import torch
import typer
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from chamelion.config import load_config
from chamelion.datasets.chamelion_dataset import ChamelionDataModule
from chamelion.training.checkpoint_io import SafeCheckpointIO
from chamelion.training.module import TrainingModule
from chamelion.utils.seed import set_seed


def train(
    config: Path = typer.Option(
        Path("config/cham.yaml"),
        "--config",
        exists=True,
        help="Path to the training configuration file",
    ),
    data: Optional[Path] = typer.Option(
        None,
        "--data",
        help="Override paths.data from the configuration",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Override paths.cache from the configuration",
    ),
    logs_dir: Optional[Path] = typer.Option(
        None,
        "--logs-dir",
        help="Override paths.logs from the configuration",
    ),
    max_epochs: Optional[int] = typer.Option(
        None,
        "--max-epochs",
        min=1,
        help="Override training.max_epochs",
    ),
    resume: Optional[Path] = typer.Option(
        None,
        "--resume",
        exists=True,
        dir_okay=False,
        help="Resume full training state from a tensor/primitive-only Lightning .ckpt",
    ),
):
    cfg = load_config(config)
    data_dir = Path(data or cfg.paths.data).expanduser()
    resolved_cache_dir = cache_dir if cache_dir is not None else cfg.paths.cache
    resolved_cache_dir = (
        Path(resolved_cache_dir).expanduser() if resolved_cache_dir is not None else None
    )
    resolved_logs_dir = Path(logs_dir or cfg.paths.logs).expanduser()

    if not data_dir.is_dir():
        raise typer.BadParameter(f"Dataset directory does not exist: {data_dir}")

    if resolved_cache_dir is not None:
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_logs_dir.mkdir(parents=True, exist_ok=True)

    if max_epochs is not None:
        cfg.training.max_epochs = max_epochs

    set_seed(cfg.training.seed)
    model = TrainingModule(cfg)

    dataset = ChamelionDataModule(
        dataloader=cfg.training.dataloader,
        data_dir=data_dir,
        config=cfg,
        cache_dir=resolved_cache_dir,
    )

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    scan_checkpoint_saver = ModelCheckpoint(
        monitor="val_moving_iou_scan",
        filename=cfg.training.id + "_{epoch:03d}_{val_moving_iou_scan:.3f}",
        mode="max",
        save_last=True,
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        resolved_logs_dir,
        name=cfg.training.id,
        default_hp_metric=False,
    )

    torch.set_float32_matmul_precision("high")
    trainer_options = {
        "accelerator": cfg.training.accelerator,
        "devices": cfg.training.devices,
        "logger": tb_logger,
        "max_epochs": cfg.training.max_epochs,
        "accumulate_grad_batches": cfg.training.accumulate_grad_batches,
        "precision": cfg.training.precision,
        "log_every_n_steps": cfg.training.log_every_n_steps,
        "callbacks": [lr_monitor, scan_checkpoint_saver],
        "plugins": [SafeCheckpointIO()],
    }
    typer.echo(f"Dataset: {data_dir}")
    typer.echo(f"Cache: {resolved_cache_dir or 'disabled'}")
    typer.echo(f"Logs: {resolved_logs_dir}")
    typer.echo(f"Seed: {cfg.training.seed}")

    trainer = Trainer(**trainer_options)

    if resume is not None:
        typer.echo(f"Resuming full training state: {resume}")
    trainer.fit(model, dataset, ckpt_path=str(resume) if resume is not None else None)


if __name__ == "__main__":
    typer.run(train)
