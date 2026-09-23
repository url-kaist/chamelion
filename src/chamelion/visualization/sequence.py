#!/usr/bin/env python3
"""Run historical inference sequentially with the original Polyscope controls."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from chamelion.config import load_config
from chamelion.datasets.cham import ChamDataLoader
from chamelion.inference.map_update import HistoricalMapAccumulator, historical_frame
from chamelion.inference.predict import predict_frame
from chamelion.utils.checkpoint import load_model
from chamelion.utils.utils_dyn import local_to_global
from chamelion.visualization.preview import MapUpdatePreview
from chamelion.visualization.sequential import SequentialVisualizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--profile", required=True, choices=["custom", "lista"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("/workspace/config/cham.yaml"))
    parser.add_argument(
        "--inference-config", type=Path, default=Path("/workspace/config/inference.yaml")
    )
    args = parser.parse_args()
    if Path(args.sequence).is_absolute() or ".." in Path(args.sequence).parts:
        raise ValueError("Sequence must stay inside dataset")
    cfg = load_config(args.config)
    profile = yaml.safe_load(args.inference_config.read_text())["profiles"][args.profile]
    dataset = ChamDataLoader(args.data, args.sequence, mode_test=True)
    torch.set_num_threads(4)
    torch.manual_seed(66)
    np.random.seed(66)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sequential inference")
    torch.cuda.set_per_process_memory_fraction(0.4)
    device = torch.device("cuda")
    model = load_model(args.checkpoint, cfg.mos.voxel_size_mos, device)
    state = HistoricalMapAccumulator(
        dataset.gt_map_points,
        voxel_size=profile["belief_voxel_size"],
        confidence_threshold=profile["map_confidence_threshold"],
        max_z=profile["max_z"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)  # Screenshots stay in the new writable session directory.
    viewer = SequentialVisualizer(change_thres_map=0.5)
    viewer.set_voxel_size(profile["belief_voxel_size"])
    map_preview = MapUpdatePreview(voxel_size=cfg.mos.voxel_size_mos)
    for index in range(len(dataset)):
        scan, labels, timestamp, pose = dataset[index]
        scan = local_to_global(scan[:, :3], pose)
        frame = predict_frame(
            model,
            historical_frame(
                scan,
                dataset.gt_map_points,
                pose,
                labels,
                dataset.gt_map_label,
                min_range=cfg.mos.min_range_mos,
                max_range=cfg.mos.max_range_mos,
                max_z=profile["max_z"],
            ),
            device,
        )
        frame["scan_pred"] &= frame["scan_confidence"] <= profile["scan_confidence_threshold"]
        state.update(frame["map_mask"], frame["map_confidence"])
        final_map_prediction = state.predict()
        accumulated = final_map_prediction[frame["map_mask"]].astype(float)
        map_preview.add_scan(frame["scan_xyz"], frame["scan_pred"])
        updated_points, updated_colors, removed_points = map_preview.compose(
            dataset.gt_map_points, final_map_prediction, dataset.gt_map_label < 250
        )
        viewer.set_updated_map(updated_points, updated_colors, removed_points)
        viewer.status = (
            f"{args.sequence} | {index + 1}/{len(dataset)} | timestamp {timestamp} | "
            f'scan points {len(frame["scan_xyz"])}'
        )
        print(viewer.status, flush=True)
        # No XY visibility overlay: this only affects display, never inference.
        viewer.update(
            frame["scan_xyz"],
            frame["map_xyz"],
            frame["scan_pred"],
            frame["map_pred"],
            accumulated,
            frame["map_confidence"],
            pose,
            np.zeros(len(accumulated), dtype=bool),
        )
        torch.cuda.empty_cache()
    viewer.status = f"Complete: {len(dataset)} frames. Press Q to close."
    viewer._play_mode = False
    viewer._ps.show()


if __name__ == "__main__":
    main()
