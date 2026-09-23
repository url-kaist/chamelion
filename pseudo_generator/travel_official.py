"""Official TRAVEL adapter. All returned labels retain the input point order."""
from __future__ import annotations

import numpy as np


class OfficialTravel:
    def __init__(self, config):
        import travel_seg as ts

        self.ground = ts.TravelGroundSeg(ts.GroundSegConfig(**config["ground"]))
        self.objects = ts.ObjectCluster(ts.ObjectClusterConfig(**config["objects"]))

    def segment(self, points, pose, initial_pose, preprocessing):
        xyz = np.asarray(points[:, :3], dtype=np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError("Input cloud contains NaN or infinity")
        # Retain the former orientation normalization and height filter explicitly.
        rotation = initial_pose[:3, :3].T @ pose[:3, :3]
        levelled = xyz @ rotation.T if preprocessing["normalize_orientation"] else xyz
        height = preprocessing.get("max_height")
        mask = np.ones(len(xyz), dtype=bool) if height is None else levelled[:, 2] < height
        ground = np.zeros(len(xyz), dtype=np.int32)
        instances = np.zeros(len(xyz), dtype=np.int32)
        indices = np.flatnonzero(mask)
        if not len(indices):
            raise ValueError("Preprocessing discarded every point")
        ground_mask, _ = self.ground.estimate_ground(levelled[indices])
        ground[indices] = ground_mask.astype(np.int32)
        nonground_indices = indices[~ground_mask]
        if len(nonground_indices):
            # Match the old object step's sensor-local coordinates, not levelled XYZ.
            instances[nonground_indices] = self.objects.segment_objects(xyz[nonground_indices])
        return ground, instances
