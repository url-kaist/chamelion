"""Historical map update and evaluation, extracted from b4d67f7.

Preserves update_map_prob_modify/process_final_prediction_map, not a newly
substituted Bayesian formula. Low-confidence observations do not reset state.
The hash groups neighboring map points; it is NOT a training visibility mask.
"""
import numpy as np


def historical_map_hash(points, voxel_size):
    points_xyz = np.round(np.asarray(points)[:, :3] / voxel_size).astype(np.int32)
    vec = points_xyz.view(np.uint32)
    return (vec[:, 0] * 73856093) ^ (vec[:, 1] * 19349669) ^ (vec[:, 2] * 83492791)


class HistoricalMapAccumulator:
    def __init__(self, points, *, voxel_size, confidence_threshold, max_z):
        self.points = np.asarray(points)
        self.map_hash = historical_map_hash(points, voxel_size)
        self.threshold = confidence_threshold
        self.max_z = max_z
        self.sum = np.zeros(self.map_hash.shape, dtype=np.float32)
        self.count = np.zeros(self.map_hash.shape, dtype=np.float32)

    def update(self, map_mask, confidence):
        map_mask = np.asarray(map_mask, dtype=bool)
        confidence = np.asarray(confidence, dtype=np.float64)
        if map_mask.shape != self.count.shape or confidence.shape != (int(map_mask.sum()),):
            raise ValueError("Map mask and confidence must match the fixed prior map")
        if not np.isfinite(confidence).all():
            raise ValueError("Non-finite confidence")
        selected_hash = self.map_hash[map_mask][confidence > self.threshold]
        selected = np.isin(self.map_hash, selected_hash)
        # Exact historical behavior: propagate selection within a hash cell,
        # but increment only the points present in this frame's map input.
        self.sum[selected & map_mask] += confidence[selected[map_mask]]
        self.count[selected & map_mask] += 1

    def predict(self):
        changed = self.count == 0
        changed[self.points[:, 2] > self.max_z] = False
        return changed


def historical_frame(scan, prior, pose, scan_labels, map_labels, *, min_range, max_range, max_z):
    """Original range/relative-height crop and annotated HD exclusion.

    Static/change labels are not provided as network features. The explicit
    GT-assisted HD exclusion matches the original benchmark pre-processing.
    """
    result = {"pose": pose.copy()}
    location = np.asarray(pose[:3, 3], dtype=np.float32)
    for name, points, labels in (("scan", scan, scan_labels), ("map", prior, map_labels)):
        distance = np.linalg.norm(points - location, axis=1)
        mask = distance >= min_range
        if max_range > 0:
            mask &= distance <= max_range
        mask &= points[:, 2] - location[2] <= max_z
        mask &= labels < 250
        result[f"{name}_mask"] = mask
        result[f"{name}_xyz"] = points[mask]
    # The historical network accepts a map-only input when HD removal leaves
    # an empty scan. Do not skip that frame or change its filtering conditions.
    if not len(result["scan_xyz"]) and not len(result["map_xyz"]):
        raise ValueError("Both inputs are empty after historical preprocessing")
    return result


def map_scores(predicted_change, labels):
    """Final-map PR, RR and their harmonic mean; each point counted once."""
    pred = np.asarray(predicted_change)
    labels = np.asarray(labels)
    if pred.shape != labels.shape or labels.ndim != 1 or not np.isin(pred, [0, 1]).all():
        raise ValueError("Expected one binary prediction per labeled map point")
    pred = pred.astype(bool)
    static, changed = labels == 0, np.isin(labels, [1, 2])
    tn, fp = int(np.sum(static & ~pred)), int(np.sum(static & pred))
    tp, fn = int(np.sum(changed & pred)), int(np.sum(changed & ~pred))
    pr = tn / (tn + fp) if tn + fp else None
    rr = tp / (tp + fn) if tp + fn else None
    f1 = None if pr is None or rr is None else (2 * pr * rr / (pr + rr) if pr + rr else 0.0)
    return dict(
        pr=pr,
        rr=rr,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        evaluated_points=tp + fp + fn + tn,
        ignored_points=int(len(labels) - tp - fp - fn - tn),
    )
