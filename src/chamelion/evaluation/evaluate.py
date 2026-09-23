#!/usr/bin/env python3
"""Evaluate the preserved b4d67f7 inference logic: scan IoU, final-map PR/RR/F1."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from chamelion.config import load_config
from chamelion.datasets.cham import ChamDataLoader
from chamelion.evaluation.common import (
    file_hash,
    metrics,
    plot_points,
    plt,
    preview,
    scores,
)
from chamelion.inference.map_update import (
    HistoricalMapAccumulator,
    historical_frame,
    map_scores,
)
from chamelion.inference.predict import predict_frame
from chamelion.utils.checkpoint import load_model
from chamelion.utils.utils_dyn import local_to_global
from chamelion.visualization.data import save_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--sequence", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/cham.yaml"))
    parser.add_argument("--inference-config", type=Path, default=Path("config/inference.yaml"))
    parser.add_argument("--profile", choices=["custom", "lista"], required=True)
    parser.add_argument("--save-all-frames", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory")
    cfg = load_config(args.config)
    settings = yaml.safe_load(args.inference_config.read_text())
    if settings["protocol"] != "historical_b4d67f7":
        raise ValueError("Unsupported inference protocol")
    profile = settings["profiles"][args.profile]
    train_sessions = {name.split("/")[0] for name in cfg.training.train + cfg.training.val}
    for sequence in args.sequence:
        if Path(sequence).is_absolute() or ".." in Path(sequence).parts:
            raise ValueError("Sequence must stay inside the dataset")
        if sequence.split("/")[0] in train_sessions:
            raise ValueError("Test sequence overlaps configured training/validation sessions")
    torch.set_num_threads(4)
    torch.manual_seed(66)
    np.random.seed(66)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA environment required")
    torch.cuda.set_per_process_memory_fraction(0.4)
    device = torch.device("cuda")
    model = load_model(args.checkpoint, cfg.mos.voxel_size_mos, device)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        protocol=settings["protocol"],
        profile=args.profile,
        settings=profile,
        checkpoint_sha256=file_hash(args.checkpoint),
        map_metric="PR/RR/F1 on each final prior-map point exactly once",
        map_update="Preserved update_map_prob_modify: confidence-selected hash cells and counts",
        low_confidence="Keep previous state; no reset",
        gt_usage="Annotated HD points (labels >=250) excluded before inference as in historical code; no static/change GT features",
        test_set_tuning=False,
        bayesian_formula_substituted=False,
        warning="Historical code protocol, not a claim of exact paper-score reproduction or Eq. (10) implementation",
        sequences={},
    )
    sections = []
    for sequence in args.sequence:
        dataset = ChamDataLoader(args.data, sequence, mode_test=True)
        prior, gt_map = dataset.gt_map_points, dataset.gt_map_label
        state = HistoricalMapAccumulator(
            prior,
            voxel_size=profile["belief_voxel_size"],
            confidence_threshold=profile["map_confidence_threshold"],
            max_z=profile["max_z"],
        )
        name = sequence.replace("/", "__")
        output = args.output / name
        output.mkdir()
        total = dict(tp=0, fp=0, fn=0, tn=0)
        rows, images = [], []
        selected = {0, len(dataset) // 2, len(dataset) - 1}
        print(f"{sequence}: {len(dataset)} frames, {len(prior)} prior-map points", flush=True)
        for index in range(len(dataset)):
            local_scan, labels, timestamp, pose = dataset[index]
            scan = local_to_global(local_scan[:, :3], pose)
            frame = predict_frame(
                model,
                historical_frame(
                    scan,
                    prior,
                    pose,
                    labels,
                    gt_map,
                    min_range=cfg.mos.min_range_mos,
                    max_range=cfg.mos.max_range_mos,
                    max_z=profile["max_z"],
                ),
                device,
            )
            frame["scan_pred"] &= frame["scan_confidence"] <= profile["scan_confidence_threshold"]
            state.update(frame["map_mask"], frame["map_confidence"])
            frame["map_pred"] = state.predict()[frame["map_mask"]]
            frame.update(
                scan_gt=labels[frame["scan_mask"]],
                map_gt=gt_map[frame["map_mask"]],
                timestamp=np.asarray(int(timestamp)),
                protocol=np.asarray(settings["protocol"]),
            )
            count = metrics(frame["scan_pred"], frame["scan_gt"])
            for key, value in count.items():
                total[key] += value
            rows.append(
                dict(
                    frame_index=index,
                    timestamp=int(timestamp),
                    scan_input_points=len(frame["scan_xyz"]),
                    map_input_points=len(frame["map_xyz"]),
                    **scores(count),
                )
            )
            nonempty_pair = len(frame["scan_xyz"]) > 0 and len(frame["map_xyz"]) > 0
            if index in selected and nonempty_pair:
                image_name = f"frame_{int(timestamp):06d}.png"
                preview(
                    output / image_name,
                    sequence,
                    timestamp,
                    frame["scan_xyz"],
                    frame["scan_gt"],
                    frame["scan_pred"],
                    frame["map_xyz"],
                    frame["map_gt"],
                    frame["map_pred"],
                    caption="Historical confidence filtering and accumulated map state; fixed settings",
                    prediction_title="Historical prediction",
                )
                images.append(f"{name}/{image_name}")
            if (index in selected or args.save_all_frames) and nonempty_pair:
                save_frame(output / f"frame_{int(timestamp):06d}.npz", frame)
            torch.cuda.empty_cache()
            print(f"{sequence}: {index + 1}/{len(dataset)}", flush=True)
        final_pred = state.predict()
        result = dict(
            frames=len(dataset),
            scan=scores(total),
            map=map_scores(final_pred, gt_map),
            empty_scan_frames=[r["timestamp"] for r in rows if not r["scan_input_points"]],
            empty_map_frames=[r["timestamp"] for r in rows if not r["map_input_points"]],
            map_points_ever_updated=int(np.count_nonzero(state.count)),
            previews=images,
        )
        with (output / "final_map.npz").open("xb") as stream:
            np.savez_compressed(
                stream,
                map_xyz=prior,
                map_gt=gt_map,
                map_pred=final_pred,
                confidence_sum=state.sum,
                observation_count=state.count,
                protocol=np.asarray(settings["protocol"]),
            )
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        valid = np.isin(gt_map, [0, 1, 2])
        for ax, classes, title in zip(
            axes, [gt_map > 0, final_pred], ["Ground truth", "Final accumulated map"]
        ):
            plot_points(
                ax,
                prior[valid],
                classes[valid].astype(int),
                ["#b6bcc4", "#df463d"],
                ["Static", "Negative change"],
            )
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output / "final_map.png", dpi=150)
        plt.close(fig)
        with (output / "frames.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        report["sequences"][sequence] = result
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({sequence: result}, indent=2), flush=True)

        def fmt(x):
            return "N/A" if x is None else f"{100*x:.2f}%"

        cells = "".join(
            f"<td>{fmt(x)}</td>"
            for x in [result["scan"]["iou"], *[result["map"][k] for k in ("pr", "rr", "f1")]]
        )
        sections.append(
            f"<h2>{html.escape(sequence)}</h2><table><tr><th>Scan IoU</th><th>Map PR</th><th>Map RR</th><th>Map F1</th></tr><tr>{cells}</tr></table>"
            f'<img src="{name}/final_map.png">' + "".join(f'<img src="{p}">' for p in images)
        )
        (args.output / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Chamelion evaluation</title>'
            "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto}img{width:100%}td,th{padding:12px}</style>"
            "<h1>Historical-code evaluation</h1><p>"
            + html.escape(report["warning"])
            + "</p>"
            + "".join(sections)
            + "</html>"
        )


if __name__ == "__main__":
    main()
