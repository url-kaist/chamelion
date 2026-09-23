# original code : vdbfusion
# modified by: seoyeon Jang

import glob
import os

import numpy as np

# import pcl
import open3d as o3d
import pandas as pd
from source_paths import source_clouds
from trimesh import transform_points


class Config:
    # pre-processing params
    apply_pose = True
    correct_scan = True
    min_range = 2.0
    max_range = 70.0

    is_bin = True
    is_semantic = False
    is_estimated_inst = True
    is_calib = True


class SemanticKittiDataset:
    def __init__(self, config):
        """Simple KITTI DataLoader to provide a ready-to-run example.

        Heavily inspired in PyLidar SLAM
        """
        # Config stuff
        kitti_root_dir = config["dataset_root"]
        sequence = config["seq"]

        pred_inst_dir = config["pred_inst_root"]

        self.sequence = sequence  # str(int(sequence)).zfill(2)
        self.config = Config()
        self.config.apply_pose = config["apply_pose"]
        self.config.correct_scan = config["correct_scan"]
        # self.config.min_range = config['min_range']
        # self.config.max_range = config['max_range']
        self.config.is_bin = config["is_bin"]
        self.config.is_calib = config["is_calib"]
        self.config.is_semantic = config["is_semantic"]
        self.config.is_estimated_inst = config["is_estimated_inst"]

        self.kitti_sequence_dir = os.path.join(kitti_root_dir, "sequences", self.sequence)
        self.clouds_dir = str(source_clouds(self.kitti_sequence_dir)) + os.sep
        self.labels_dir = os.path.join(self.kitti_sequence_dir, "labels/")

        label_sequence = config.get("pred_inst_sequence", self.sequence)
        self.ground_dir = os.path.join(pred_inst_dir, label_sequence, "travel_btms/")
        self.instance_dir = os.path.join(pred_inst_dir, label_sequence, "travel_aos/")
        # self.instance_dir = os.path.join(pred_inst_dir, self.sequence, "hdbscan/")
        # self.instance_dir = os.path.join(pred_inst_dir, self.sequence, "euclidian/")

        self.is_bin = self.config.is_bin
        self.is_calib = self.config.is_calib
        self.is_semantic = self.config.is_semantic
        self.is_estimated_inst = self.config.is_estimated_inst

        # Read stuff
        # self.calibration = self.read_calib_file(os.path.join(kitti_root_dir, "sequences",self.sequence, "calib.txt"))
        self.poses = self.load_poses(
            os.path.join(kitti_root_dir, "sequences", self.sequence, "poses.txt")
        )

        if self.is_bin:
            self.scan_files = sorted(glob.glob(self.clouds_dir + "*.bin"))
        else:
            self.scan_files = sorted(glob.glob(self.clouds_dir + "*.pcd"))

        self.gt_label_files = sorted(glob.glob(self.labels_dir + "*.label"))

        self.pred_inst_files = sorted(glob.glob(self.instance_dir + "*.label"))
        self.pred_ground_files = sorted(glob.glob(self.ground_dir + "*.label"))

        self.num_frames = len(self.scan_files)
        if not self.num_frames:
            raise ValueError(f"No source scans in {self.clouds_dir}")
        if len(self.poses) != self.num_frames:
            raise ValueError("Interactive generation requires one pose row for every source scan")
        from pathlib import Path

        if [int(Path(p).stem) for p in self.scan_files] != list(range(self.num_frames)):
            raise ValueError("Interactive generation requires contiguous zero-based scan filenames")
        if self.is_estimated_inst:
            names = [Path(p).stem for p in self.scan_files]
            if names != [Path(p).stem for p in self.pred_inst_files] or names != [
                Path(p).stem for p in self.pred_ground_files
            ]:
                raise ValueError(
                    "Missing or mismatched TRAVEL labels; run label_instances.py first"
                )

    def __getitem__(self, idx):
        if self.is_estimated_inst:
            inst_label, ground_label = self.inst_ground_labels(idx)
            # inst_label = inst_label >> 16
            return self.scans(idx), inst_label, ground_label, self.poses[idx]

        elif self.is_semantic:
            return self.scans(idx), self.labels(idx), self.poses[idx]
        else:
            return self.scans(idx), self.poses[idx]

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        if self.is_bin:
            return self.read_point_cloud_bin(idx, self.scan_files[idx], self.config)
        else:
            return self.read_point_cloud(idx, self.scan_files[idx], self.config)

    def labels(self, idx):
        return np.fromfile(self.gt_label_files[idx], dtype=np.uint32).reshape((-1))

    def inst_ground_labels(self, idx):
        inst_label = np.fromfile(self.pred_inst_files[idx], dtype=np.uint32).reshape((-1))
        ground_label = np.fromfile(self.pred_ground_files[idx], dtype=np.uint32).reshape((-1))

        return inst_label, ground_label

    def read_point_cloud(self, idx: int, scan_file: str, config: Config):
        points = o3d.io.read_point_cloud(scan_file).points
        points = np.asarray(points)
        points = points[:, :3]
        points = points[np.linalg.norm(points, axis=1) <= config.max_range]
        points = points[np.linalg.norm(points, axis=1) >= config.min_range]
        points = transform_points(points, self.poses[idx]) if config.apply_pose else points
        return points

    def read_point_cloud_bin(self, idx: int, scan_file: str, config: Config):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        points = self._correct_scan(points) if config.correct_scan else points[:, :3]
        # points = points[np.linalg.norm(points, axis=1) <= config.max_range]
        # points = points[np.linalg.norm(points, axis=1) >= config.min_range]
        points = transform_points(points, self.poses[idx]) if config.apply_pose else points

        return points

    @staticmethod
    def _correct_scan(scan: np.ndarray):
        """Corrects the calibration of KITTI's HDL-64 scan.

        Taken from PyLidar SLAM
        """
        xyz = scan[:, :3]
        n = scan.shape[0]
        z = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        axes = np.cross(xyz, z)
        # Normalize the axes
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        theta = 0.205 * np.pi / 180.0

        # Build the rotation matrix for each point
        c = np.cos(theta)
        s = np.sin(theta)

        u_outer = axes.reshape(n, 3, 1) * axes.reshape(n, 1, 3)
        u_cross = np.zeros((n, 3, 3), dtype=np.float32)
        u_cross[:, 0, 1] = -axes[:, 2]
        u_cross[:, 1, 0] = axes[:, 2]
        u_cross[:, 0, 2] = axes[:, 1]
        u_cross[:, 2, 0] = -axes[:, 1]
        u_cross[:, 1, 2] = -axes[:, 0]
        u_cross[:, 2, 1] = axes[:, 0]

        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        rotations = c * eye + s * u_cross + (1 - c) * u_outer
        corrected_scan = np.einsum("nij,nj->ni", rotations, xyz)
        return corrected_scan

    def load_poses(self, pose_path):
        """Load ground truth poses (T_w_cam0) from file.
        Args:
        pose_path: (Complete) filename for the pose file
        Returns:
        A numpy array of size nx4x4 with n poses as 4x4 transformation
        matrices
        """
        # Read and parse the poses
        poses = []
        try:
            if ".txt" in pose_path:
                with open(pose_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                        poses.append(T_w_cam0)
            else:
                poses = np.load(pose_path)["arr_0"]

        except FileNotFoundError:
            print("Ground truth poses are not avaialble.")

        return np.array(poses)

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)
                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict


class HeLiMOSDataset(SemanticKittiDataset):
    def __init__(self, config):
        super().__init__(config)
        kitti_root_dir = config[["build_dataset"]]["dataset_root"]
        sequence = config["seq"]

        self.sequence = sequence
        self.config = Config()
        self.kitti_sequence_dir = os.path.join(kitti_root_dir, self.sequence)
        self.clouds_dir = str(source_clouds(self.kitti_sequence_dir)) + os.sep
        self.labels_dir = os.path.join(self.kitti_sequence_dir, "labels/")
        self.is_bin = self.config.is_bin

        # Read stuff
        # self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))
        self.poses = self.load_poses(
            os.path.join(kitti_root_dir, "sequences", self.sequence, "poses.txt")
        )
        if self.is_bin:
            self.scan_files = sorted(glob.glob(self.clouds_dir + "*.bin"))
        else:
            self.scan_files = sorted(glob.glob(self.clouds_dir + "*.pcd"))

        self.gt_label_files = sorted(glob.glob(self.labels_dir + "*.label"))
