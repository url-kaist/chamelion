"""Viewer data and safe output handling; no model or GUI imports."""
from pathlib import Path

import numpy as np


def validate_frame(frame):
    for name in ("scan", "map"):
        points = np.asarray(frame[f"{name}_xyz"])
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"{name}: expected nonempty finite Nx3 points")
        for suffix in ("gt", "pred", "logits", "confidence"):
            key = f"{name}_{suffix}"
            if key in frame:
                values = np.asarray(frame[key])
                if values.shape != (len(points),) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {key} array")
                if suffix == "pred" and not np.isin(values, [0, 1]).all():
                    raise ValueError(f"{key} must contain binary predictions")
    if "pose" in frame:
        pose = frame["pose"]
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Invalid pose")
    return frame


def point_colors(frame, name, mode):
    colors = np.tile([0.62, 0.66, 0.70], (len(frame[f"{name}_xyz"]), 1))
    change_color = [0.10, 0.52, 0.95] if name == "scan" else [0.95, 0.23, 0.20]
    if mode == "Prediction" and f"{name}_pred" in frame:
        colors[frame[f"{name}_pred"].astype(bool)] = change_color
    elif mode in ("Ground truth", "Errors") and f"{name}_gt" in frame:
        gt = frame[f"{name}_gt"]
        valid = np.isin(gt, [0, 1, 2])
        target = gt > 0
        if mode == "Errors" and f"{name}_pred" in frame:
            pred = frame[f"{name}_pred"].astype(bool)
            colors[valid & target & pred] = [0.1, 0.75, 0.35]
            colors[valid & ~target & pred] = [0.95, 0.23, 0.20]
            colors[valid & target & ~pred] = [0.10, 0.52, 0.95]
        elif mode == "Ground truth":
            colors[valid & target] = change_color
        colors[~valid] = [0.25, 0.25, 0.25]
    return colors


def save_frame(path, frame):
    validate_frame(frame)
    # Exclusive creation: no existing result, source or checkpoint is overwritten.
    with Path(path).open("xb") as stream:
        np.savez_compressed(
            stream, **{key: value for key, value in frame.items() if not key.endswith("_mask")}
        )


class SavedFrames:
    def __init__(self, path):
        path = Path(path)
        self.files = [path] if path.is_file() else sorted(path.rglob("frame_*.npz"))
        if not self.files:
            raise ValueError("No frame_*.npz results found")

    def __len__(self):
        return len(self.files)

    def load(self, index):
        path = self.files[index]
        with np.load(path, allow_pickle=False) as archive:
            frame = {key: archive[key] for key in archive.files}
        return validate_frame(frame), f"{path.parent.name}/{path.name}"


class CloudFrames:
    """Label-free PCD input. Numeric scan stems index the full poses table."""

    def __init__(self, prior, scans, poses, coordinates, min_range, max_range):
        import open3d as o3d

        self.read_cloud = lambda path: np.asarray(o3d.io.read_point_cloud(str(path)).points)
        self.prior = self.read_cloud(prior)
        self.files = sorted(Path(scans).glob("*.pcd"))
        if not self.files or any(not path.stem.isdigit() for path in self.files):
            raise ValueError("Expected numeric PCD scan filenames")
        self.files.sort(key=lambda path: int(path.stem))
        if len({int(path.stem) for path in self.files}) != len(self.files):
            raise ValueError("Duplicate numeric frame IDs")
        rows = np.loadtxt(poses, ndmin=2)
        if rows.shape[1] != 12 or not np.isfinite(rows).all():
            raise ValueError("poses.txt must have 12 finite numbers per row")
        self.poses = np.tile(np.eye(4), (len(rows), 1, 1))
        self.poses[:, :3] = rows.reshape(-1, 3, 4)
        if int(self.files[-1].stem) >= len(rows):
            raise ValueError("A scan ID exceeds the poses table; do not renumber subsets")
        if coordinates not in ("global", "local"):
            raise ValueError("Choose global or local scan coordinates")
        self.coordinates, self.min_range, self.max_range = coordinates, min_range, max_range

    def __len__(self):
        return len(self.files)

    def load(self, index):
        from chamelion.inference.predict import prepare_frame

        path = self.files[index]
        pose = self.poses[int(path.stem)]
        scan = self.read_cloud(path)
        if self.coordinates == "local":
            scan = scan @ pose[:3, :3].T + pose[:3, 3]
        frame = prepare_frame(scan, self.prior, pose, self.min_range, self.max_range)
        frame["timestamp"] = np.asarray(int(path.stem))
        return frame, path.stem
