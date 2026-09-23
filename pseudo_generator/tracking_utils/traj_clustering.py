import argparse

import numpy as np


class Tile:
    def __init__(self):
        self.i = 0
        self.j = 0
        self.k = 0
        self.indexes = []
        self.x = 0
        self.y = 0
        self.z = 0
        self.size = 0


class trajectory_clustering:
    def __init__(self, arg):
        self.trajectory = None

        self.max_distance = arg.max_distance
        self.min_distance = arg.min_distance

        self.tile_size = arg.tile_size

        self.min_x = 0.0
        self.max_x = 0.0

        self.min_y = 0.0
        self.max_y = 0.0

        self.offset_ = np.zeros(3)
        self.num_tiles_ = np.zeros(3)

        self.poses_ = self.load_poses(arg.pose_path)
        self.cluster_indices_ = []

        self.tiles_ = []

    def load_poses(self, pose_path):
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

    def cluster_trajectory(self):
        poses_ = self.poses_

        for i in range(len(poses_)):
            aligned_pose = np.linalg.inv(poses_[0]) @ self.poses_[i]
            trans = aligned_pose[:3, 3]
