from pathlib import Path

import numpy as np
import open3d as o3d

from chamelion.utils.utils_dyn import local_to_global


class ChamDataLoader:
    """Load a Chamelion sequence from either a dataset root or its sequences directory."""

    def __init__(
        self,
        data_dir: str | Path,
        sequence: str,
        use_xyzi: bool = False,
        mode_test: bool = True,
        *_,
        **__,
    ):
        self.mode_test = mode_test
        self.sequence_id = sequence
        self.use_xyzi = use_xyzi

        data_path = Path(data_dir).expanduser()
        if (data_path / "sequences").is_dir():
            self.dataset_root = data_path
            self.sequences_root = data_path / "sequences"
        elif data_path.name == "sequences" and data_path.is_dir():
            self.dataset_root = data_path.parent
            self.sequences_root = data_path
        else:
            raise FileNotFoundError(
                f"Expected a dataset root containing 'sequences' or the sequences directory: "
                f"{data_path}"
            )

        sequence_root = self.sequences_root / sequence
        public_layout = (sequence_root / "prior_map.pcd").is_file()
        if public_layout:
            self.sequence_dir = sequence_root / "scans"
            self.gt_scan_label_path = sequence_root / "scan_labels"
            map_dir = sequence_root
            map_label_dir = sequence_root / "map_labels"
            pose_file = self.sequences_root / sequence.split("/", maxsplit=1)[0] / "poses.txt"
        elif self.mode_test:
            cluster_root = sequence_root / "toss" / "cluster_test"
            self.sequence_dir = cluster_root / "test"
            self.gt_scan_label_path = self.dataset_root / "gt_labels" / sequence / "scans"
            map_dir = cluster_root / "map"
            map_label_dir = self.dataset_root / "gt_labels" / sequence / "maps"
            pose_file = sequence_root / "toss" / "poses.txt"
        else:
            self.sequence_dir = sequence_root / "scans"
            self.gt_scan_label_path = sequence_root / "scans_label"
            map_dir = sequence_root / "map"
            map_label_dir = self._find_map_label_dir(sequence_root)
            base_sequence = sequence.split("/", maxsplit=1)[0]
            pose_file = self.sequences_root / base_sequence / "poses.txt"

        self._require_directory(self.sequence_dir, "scan directory")
        self._require_directory(self.gt_scan_label_path, "scan-label directory")
        self._require_directory(map_dir, "map directory")
        self._require_directory(map_label_dir, "map-label directory")

        self.prior_map_path = (
            sequence_root / "prior_map.pcd"
            if public_layout
            else self._first_file(map_dir, "prior map")
        )
        map_cloud = o3d.io.read_point_cloud(str(self.prior_map_path))
        self.gt_map_points = np.asarray(map_cloud.points)

        scan_paths = self._indexed_files(self.sequence_dir)
        scan_label_paths = self._indexed_files(self.gt_scan_label_path)
        map_label_paths = (
            self._files(map_label_dir)
            if self.mode_test or public_layout
            else self._indexed_files(map_label_dir)
        )
        if public_layout and len(map_label_paths) > 1:
            map_label_paths = self._indexed_files(map_label_dir)
        scan_names = [path.stem for path in scan_paths]
        label_names = [path.stem for path in scan_label_paths]
        if scan_names != label_names:
            raise ValueError(f"Scan and scan-label filenames do not match for {sequence}")

        map_label_names = [path.stem for path in map_label_paths]
        if not self.mode_test and not public_layout and scan_names != map_label_names:
            raise ValueError(f"Scan and map-label filenames do not match for {sequence}")
        if (self.mode_test or public_layout) and len(map_label_paths) not in (1, len(scan_paths)):
            raise ValueError(
                f"Expected one map label or one per scan for {sequence}, "
                f"got {len(map_label_paths)} labels for {len(scan_paths)} scans"
            )

        expected_map_label_bytes = len(self.gt_map_points) * np.dtype(np.int32).itemsize
        for map_label_path in map_label_paths:
            if map_label_path.stat().st_size != expected_map_label_bytes:
                raise ValueError(
                    f"Map point/label count mismatch at {map_label_path}: "
                    f"expected {len(self.gt_map_points)} int32 labels"
                )

        scan_timestamps = [int(path.stem) for path in scan_paths]
        if len(map_label_paths) == 1:
            self.map_label_files = {timestamp: map_label_paths[0] for timestamp in scan_timestamps}
        else:
            map_label_timestamps = [int(path.stem) for path in map_label_paths]
            if scan_timestamps != map_label_timestamps:
                raise ValueError(f"Scan and map-label timestamps do not match for {sequence}")
            self.map_label_files = dict(zip(map_label_timestamps, map_label_paths))

        # One canonical map target is shared across frames. The release trainer
        # copies it before class conversion and does not add visibility masking.
        self.gt_map_label_path = map_label_paths[0]
        self.gt_map_label = self.get_map_label(scan_timestamps[0])

        poses = self.load_poses(pose_file)
        if public_layout:
            if any(index < 0 or index >= len(poses) for index in scan_timestamps):
                raise ValueError(f"Scan frame index is outside poses.txt for {sequence}")
            poses = poses[scan_timestamps]
        if len(scan_paths) > len(poses):
            raise ValueError(
                f"More scans than poses for {sequence}: {len(scan_paths)} vs {len(poses)}"
            )
        if len(scan_paths) != len(poses):
            try:
                poses = poses[scan_timestamps]
            except IndexError as error:
                raise ValueError(
                    f"A scan timestamp is outside the pose array for {sequence}"
                ) from error

        self.gt_poses = dict(zip(scan_timestamps, poses))
        self.scan_files = dict(zip(scan_timestamps, scan_paths))
        self.scan_label_files = dict(zip(scan_timestamps, scan_label_paths))
        self.synced_timestamps = list(self.gt_poses)

    @staticmethod
    def _require_directory(path: Path, description: str) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {description}: {path}")

    @staticmethod
    def _first_file(directory: Path, description: str) -> Path:
        files = ChamDataLoader._files(directory)
        if not files:
            raise FileNotFoundError(f"No {description} file found in {directory}")
        return files[0]

    @staticmethod
    def _files(directory: Path) -> list[Path]:
        return sorted(path for path in directory.iterdir() if path.is_file())

    @classmethod
    def _indexed_files(cls, directory: Path) -> list[Path]:
        files = cls._files(directory)
        if not files:
            raise FileNotFoundError(f"No indexed files found in {directory}")
        try:
            return sorted(files, key=lambda path: int(path.stem))
        except ValueError as error:
            raise ValueError(f"Expected numeric filenames in {directory}") from error

    @staticmethod
    def _find_map_label_dir(sequence_root: Path) -> Path:
        # Legacy exports contain an auxiliary map_labels directory. map_label_ is
        # the legacy spelling of the frame-level change labels used by cluster_0.
        for name in ("map_label", "map_label_", "map_labels"):
            candidate = sequence_root / name
            if candidate.is_dir():
                return candidate
        return sequence_root / "map_label"

    def get_map_label(self, timestamp: int) -> np.ndarray:
        try:
            label_path = self.map_label_files[timestamp]
        except KeyError as error:
            raise KeyError(
                f"No map label for timestamp {timestamp} in {self.sequence_id}"
            ) from error
        return np.fromfile(label_path, dtype=np.int32).reshape(-1)

    def __len__(self) -> int:
        return len(self.synced_timestamps)

    def __getitem__(self, idx: int):
        timestamp = self.synced_timestamps[idx]
        scan_path = self.scan_files[timestamp]
        pose = self.gt_poses[timestamp]

        if self.use_xyzi:
            local_scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        else:
            scan_cloud = o3d.io.read_point_cloud(str(scan_path))
            global_scan = np.asarray(scan_cloud.points)
            local_scan = local_to_global(global_scan, np.linalg.inv(pose))

        scan_label = np.fromfile(self.scan_label_files[timestamp], dtype=np.int32).reshape(-1)
        if len(local_scan) != len(scan_label):
            raise ValueError(
                f"Scan point/label count mismatch at {scan_path}: "
                f"{len(local_scan)} points vs {len(scan_label)} labels"
            )

        return local_scan, scan_label, timestamp, pose

    @staticmethod
    def load_poses(pose_path: Path) -> np.ndarray:
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing pose file: {pose_path}")

        if pose_path.suffix == ".txt":
            poses = []
            with pose_path.open() as pose_file:
                for line_number, line in enumerate(pose_file, start=1):
                    values = np.fromstring(line, dtype=float, sep=" ")
                    if values.size != 12:
                        raise ValueError(
                            f"Expected 12 pose values at {pose_path}:{line_number}, "
                            f"got {values.size}"
                        )
                    pose = values.reshape(3, 4)
                    poses.append(np.vstack((pose, [0, 0, 0, 1])))
            return np.asarray(poses)

        return np.load(pose_path)["arr_0"]
