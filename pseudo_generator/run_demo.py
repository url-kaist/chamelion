#!/usr/bin/env python3
"""Reproduce a small pseudo dataset using only clouds, poses and official TRAVEL."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from generate_changes import low_dynamic_object_database
from travel_official import OfficialTravel

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_cloud(path, cloud_format):
    if cloud_format.startswith("bin_"):
        columns = {"bin_xyzi": 4, "bin_xyz": 3}[cloud_format]
        points = np.fromfile(path, dtype=np.float32).reshape(-1, columns)[:, :3]
    else:
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"Empty or non-finite cloud: {path}")
    return points


def point_cloud(points, color=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        cloud.paint_uniform_color(color)
    return cloud


def transform(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def choose_object(points, labels, config):
    candidates = []
    for label in np.unique(labels):
        if label == 0:
            continue
        selected = points[labels == label]
        if (
            len(selected) >= config["object_min_points"]
            and np.ptp(selected, axis=0).max() <= config["object_max_extent"]
        ):
            candidates.append((int(label), selected))
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    rank = config["object_rank"]
    if rank < 0 or rank >= len(candidates):
        raise ValueError(
            f"No eligible object at rank {rank}; found {len(candidates)}. Inspect TRAVEL labels or adjust object filters."
        )
    return candidates[rank], len(candidates)


def validate_output(dataset, sequence, frame_ids):
    submap = dataset / "sequences" / sequence / "submaps/submap_000"
    map_points = np.asarray(o3d.io.read_point_cloud(str(submap / "prior_map.pcd")).points)
    map_labels = np.fromfile(submap / "map_labels/static.label", dtype=np.int32)
    if len(map_labels) != len(map_points) or not np.isfinite(map_points).all():
        raise ValueError("Map points and labels do not match")
    if not set(np.unique(map_labels)) <= {0, 2} or not np.any(map_labels == 2):
        raise ValueError("Expected static and removed-change map labels")
    stats = []
    for index in frame_ids:
        points = np.asarray(
            o3d.io.read_point_cloud(str(submap / "scans" / f"{index:06d}.pcd")).points
        )
        labels = np.fromfile(submap / "scan_labels" / f"{index:06d}.label", dtype=np.int32)
        if len(points) != len(labels) or not np.isfinite(points).all():
            raise ValueError(f"Scan/label mismatch at frame {index}")
        if not set(np.unique(labels)) <= {0, 1} or not np.any(labels == 1):
            raise ValueError(f"No added change at frame {index}")
        stats.append({"frame": index, "points": len(points), "added": int(np.sum(labels == 1))})
    return {
        "frames": stats,
        "map_points": len(map_points),
        "removed_map_points": int(np.sum(map_labels == 2)),
    }


def preview(dataset, sequence, frame, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    submap = dataset / "sequences" / sequence / "submaps/submap_000"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, label_path, title in [
        (axes[0], "prior_map.pcd", "map_labels/static.label", "Prior map: removed points in red"),
        (
            axes[1],
            f"scans/{frame:06d}.pcd",
            f"scan_labels/{frame:06d}.label",
            "Scan: added points in blue",
        ),
    ]:
        points = np.asarray(o3d.io.read_point_cloud(str(submap / name)).points)
        labels = np.fromfile(submap / label_path, dtype=np.int32)
        static = points[labels == 0][::5]
        changed = points[labels > 0]
        ax.scatter(static[:, 0], static[:, 1], s=1, c="#b0b7bd", alpha=0.35)
        ax.scatter(changed[:, 0], changed[:, 1], s=3, c="#e44a42" if ax is axes[0] else "#2869d8")
        ax.set(title=title, xlabel="Global X (m)", ylabel="Global Y (m)")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def run(clouds, pose_path, output, config_path):
    clouds, pose_path, output = (
        Path(clouds).resolve(),
        Path(pose_path).resolve(),
        Path(output).resolve(),
    )
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    with Path(config_path).open() as stream:
        config = yaml.safe_load(stream)
    inp, generation = config["input"], config["generation"]
    sequence = generation["sequence"]
    if not re.fullmatch(r"sequence_\d+", sequence):
        raise ValueError("Invalid public sequence name")
    if inp["frames"] <= 0 or inp["stride"] <= 0 or inp["start"] < 0:
        raise ValueError("Frame count and stride must be positive; start must be nonnegative")
    if generation["map_voxel_size"] <= 0:
        raise ValueError("map_voxel_size must be positive")
    pose_values = np.loadtxt(pose_path, ndmin=2)
    if pose_values.shape[1] != 12 or not np.isfinite(pose_values).all():
        raise ValueError("poses.txt must contain 12 finite values per row")
    poses = np.tile(np.eye(4), (len(pose_values), 1, 1))
    poses[:, :3] = pose_values.reshape(-1, 3, 4)
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("Pose rotations must be orthonormal")
    suffix = ".bin" if inp["cloud_format"].startswith("bin_") else "." + inp["cloud_format"]
    available = {int(p.stem): p for p in clouds.glob("*" + suffix) if p.stem.isdigit()}
    frames = list(range(inp["start"], inp["start"] + inp["frames"] * inp["stride"], inp["stride"]))
    for frame in frames:
        if frame not in available or frame >= len(poses):
            raise ValueError(f"Missing cloud or pose at frame {frame}")
    # Native build identity is recorded by the provided Docker build.
    revision = (ROOT / "third_party/TRAVEL_REVISION").read_text().strip()
    runtime_revision = Path("/opt/TRAVEL_REVISION")
    if not runtime_revision.is_file() or runtime_revision.read_text().strip() != revision:
        raise RuntimeError(
            "Use the pinned pseudo Docker image; build it with scripts/build.sh pseudo"
        )
    segmenter = OfficialTravel(config["travel"])
    import travel_seg

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pseudo-demo-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        dataset = stage / "Const_pseudo_dataset"
        artifacts = stage / "preparation"
        for folder in ("ground_labels", "instance_labels"):
            (artifacts / folder).mkdir(parents=True)
        globals_by_frame, segmentation = {}, []
        input_hashes = {"poses.txt": digest(pose_path)}
        for frame in frames:
            path = available[frame]
            points = read_cloud(path, inp["cloud_format"])
            ground, instances = segmenter.segment(
                points, poses[frame], poses[0], config["preprocessing"]
            )
            ground.tofile(artifacts / "ground_labels" / f"{frame:06d}.label")
            instances.tofile(artifacts / "instance_labels" / f"{frame:06d}.label")
            world = transform(points, poses[frame])
            globals_by_frame[frame] = world
            if frame == frames[0]:
                (object_id, object_points), candidate_count = choose_object(
                    world, instances, generation
                )
            input_hashes[path.name] = digest(path)
            segmentation.append(
                {
                    "frame": frame,
                    "points": len(points),
                    "ground": int(ground.sum()),
                    "instances": int(len(np.unique(instances[instances > 0]))),
                }
            )
        prior = point_cloud(np.vstack(list(globals_by_frame.values()))).voxel_down_sample(
            generation["map_voxel_size"]
        )
        prior_points = np.asarray(prior.points)
        order = np.lexsort((prior_points[:, 2], prior_points[:, 1], prior_points[:, 0]))
        prior = point_cloud(prior_points[order])
        map_path = artifacts / "prior_map.pcd"
        if not o3d.io.write_point_cloud(str(map_path), prior):
            raise OSError("Failed to save prepared map")
        sequence_dir = dataset / "sequences" / sequence
        sequence_dir.mkdir(parents=True)
        shutil.copy2(pose_path, sequence_dir / "poses.txt")
        # Reuse the existing fixed-label writer, with recorded placements replacing GUI picks.
        writer = low_dynamic_object_database.__new__(low_dynamic_object_database)
        writer.save_scan_path = writer.save_map_path = str(sequence_dir / "submaps")
        writer.map_pcd_path = str(map_path)
        writer.chunk_scan = {i: point_cloud(points) for i, points in globals_by_frame.items()}
        writer.positive_dynamics = [
            point_cloud(object_points + generation["added_offset"], [0, 0, 1])
        ]
        writer.negative_dynamics = [
            point_cloud(object_points + generation["removed_offset"], [1, 0, 0])
        ]
        writer.octree_check_flag = False
        writer.save_pd_nd_added_scans_and_map(0, frames)
        checks = validate_output(dataset, sequence, frames)
        (dataset / "splits").mkdir()
        (dataset / "splits/train.txt").write_text(f"{sequence}/submaps/submap_000\n")
        (dataset / "README.md").write_text(
            "Generation demonstration only. One submap, no validation split or training run. See ../manifest.json.\n"
        )
        preview(dataset, sequence, frames[0], stage / "preview.png")
        hashes = {
            str(p.relative_to(dataset)): digest(p)
            for p in sorted(dataset.rglob("*"))
            if p.is_file()
        }
        # Verify that every source input still has the same bytes after generation.
        if digest(pose_path) != input_hashes["poses.txt"] or any(
            digest(available[i]) != input_hashes[available[i].name] for i in frames
        ):
            raise RuntimeError("Source inputs changed during generation")
        manifest = {
            "travel_commit": revision,
            "travel_package_version": travel_seg.__version__,
            "config": config,
            "resolved_ground_config": asdict(segmenter.ground.config),
            "resolved_object_config": asdict(segmenter.objects.config),
            "inputs_sha256": input_hashes,
            "dataset_sha256": hashes,
            "segmentation": segmentation,
            "selected_object": {
                "frame": frames[0],
                "id": object_id,
                "points": len(object_points),
                "eligible_candidates": candidate_count,
            },
            "validation": checks,
            "source_inputs_unchanged": True,
            "training_executed": False,
            "limitations": [
                "Official TGS, not the former custom TGS Plus.",
                "Fixed-placement demo on a short scene; no dynamic-object tracking/removal.",
                "No occlusion or sensor-return simulation; new objects are appended to each scan.",
                "Not a reproduction of historical training labels.",
            ],
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(output)
    print(json.dumps({"output": str(output), **checks}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config/travel_demo.yaml"
    )
    args = parser.parse_args()
    run(args.clouds, args.poses, args.output, args.config)
