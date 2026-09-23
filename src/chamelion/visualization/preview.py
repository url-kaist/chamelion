"""Display-only map composition; never feeds back into inference or evaluation."""
import numpy as np


class MapUpdatePreview:
    def __init__(self, voxel_size=0.1, max_added_points=250000):
        self.voxel_size = voxel_size
        self.max_added_points = max_added_points
        self.added = np.empty((0, 3), dtype=np.float64)

    def add_scan(self, points, positive_change):
        new = np.asarray(points)[np.asarray(positive_change, dtype=bool)]
        combined = np.vstack([self.added, new])
        if not len(combined):
            return
        # Display sampling only. Keep at most one positive-change point per cell.
        _, indices = np.unique(
            np.floor(combined / self.voxel_size).astype(np.int64), axis=0, return_index=True
        )
        indices = np.sort(indices)
        if len(indices) > self.max_added_points:
            indices = indices[np.linspace(0, len(indices) - 1, self.max_added_points, dtype=int)]
        self.added = combined[indices]

    def compose(self, prior, negative_change, valid):
        prior = np.asarray(prior)
        negative_change, valid = np.asarray(negative_change, bool), np.asarray(valid, bool)
        kept = prior[valid & ~negative_change]
        points = np.vstack([kept, self.added])
        colors = np.vstack(
            [
                np.tile([0.65, 0.65, 0.65], (len(kept), 1)),
                np.tile([0.1, 0.5, 1.0], (len(self.added), 1)),
            ]
        )
        return points, colors, prior[valid & negative_change]
