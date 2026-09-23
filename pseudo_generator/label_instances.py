"""Write official TRAVEL labels without changing source data."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from interactive_ui import GROUND_COLOR, MAP_COLOR, review, stage
from run_demo import read_cloud
from source_paths import source_clouds
from tqdm import tqdm
from travel_official import OfficialTravel


def preview_labels(points, ground, instances, frame):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = np.tile(MAP_COLOR, (len(points), 1))
    colors[ground == 1] = GROUND_COLOR
    palette = np.array([[0.2, 0.65, 1.0], [1.0, 0.7, 0.2], [0.75, 0.45, 0.95], [0.1, 0.85, 0.8]])
    mask = instances > 0
    colors[mask] = palette[(instances[mask] - 1) % len(palette)]
    cloud.colors = o3d.utility.Vector3dVector(colors)
    review(
        [cloud],
        f"1/5 Segmentation | Frame {frame}",
        "Green: ground | Colors: object clusters | Gray: other",
        {"C": "continue batch"},
    )


def label_instances(config_path, preview=False):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    travel_config = yaml.safe_load((config_path.parent / config["travel_config"]).read_text())
    source = config["load_dataset"]
    sequence = source["seq"]
    if Path(sequence).name != sequence or sequence in (".", ".."):
        raise ValueError("Source sequence must be a single directory name")
    source_dir = Path(source["dataset_root"]) / "sequences" / sequence
    label_sequence = source.get("pred_inst_sequence", sequence)
    if Path(label_sequence).name != label_sequence or label_sequence in (".", ".."):
        raise ValueError("Label sequence must be a single directory name")
    destination = Path(source["pred_inst_root"]) / label_sequence
    if destination.exists():
        raise FileExistsError(f"Use a new prediction directory: {destination}")
    values = np.loadtxt(source_dir / "poses.txt", ndmin=2)
    if values.shape[1] != 12 or not np.isfinite(values).all():
        raise ValueError("Expected 12 finite pose values per row")
    poses = np.tile(np.eye(4), (len(values), 1, 1))
    poses[:, :3] = values.reshape(-1, 3, 4)
    fmt = "bin_xyzi" if source["is_bin"] else "pcd"
    paths = sorted(
        source_clouds(source_dir).glob("*.bin" if source["is_bin"] else "*.pcd"),
        key=lambda p: int(p.stem),
    )
    if not paths:
        raise ValueError("No source scans found")
    travel = OfficialTravel(travel_config["travel"])
    stage(
        1,
        "Ground Segmentation and Object Clustering",
        "TRAVEL separates ground points, then clusters non-ground points into candidate objects. Tracking runs in prepare_objects.py.",
    )
    print(f"Input: {source_dir}\nNew labels: {destination}\nFrames: {len(paths)}", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="travel-labels-", dir=destination.parent) as temporary:
        staging_dir = Path(temporary) / "labels"
        for directory in ("travel_aos", "travel_btms"):
            (staging_dir / directory).mkdir(parents=True)
        progress = tqdm(paths, desc="TRAVEL ground + objects", unit="frame")
        for index, path in enumerate(progress):
            frame = int(path.stem)
            if frame < 0 or frame >= len(poses):
                raise ValueError(f"Missing pose for {path.name}")
            points = read_cloud(path, fmt)
            ground, instances = travel.segment(
                points, poses[frame], poses[0], travel_config["preprocessing"]
            )
            progress.set_postfix(
                ground=int(np.count_nonzero(ground)),
                objects=len(np.unique(instances[instances > 0])),
            )
            if preview and index == 0:
                preview_labels(points, ground, instances, frame)
            ground.tofile(staging_dir / "travel_btms" / f"{frame:06d}.label")
            instances.tofile(staging_dir / "travel_aos" / f"{frame:06d}.label")
        staging_dir.rename(destination)
    print(f"Labeled {len(paths)} scans in {destination}")
    print(f"Next: python3 pseudo_generator/prepare_objects.py {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", type=Path, default=Path(__file__).parent / "config/generator.yaml"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip the first-frame GUI review for headless runs",
    )
    args = parser.parse_args()
    label_instances(args.config, preview=not args.no_preview)
