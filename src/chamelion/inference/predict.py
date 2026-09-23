"""Fixed-pose, raw framewise inference. No visibility mask or temporal fusion."""
import numpy as np

PROTOCOL = "raw_framewise_global_coordinates"


def prepare_frame(scan_global, prior_global, pose, min_range, max_range):
    pose = np.asarray(pose)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4x4 local-to-global pose")
    if min_range < 0 or (max_range > 0 and max_range < min_range):
        raise ValueError("Invalid distance range")
    result = {"pose": pose.copy()}
    for name, points in (("scan", scan_global), ("map", prior_global)):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"{name}: expected finite Nx3 points")
        distance = np.linalg.norm(points - pose[:3, 3], axis=1)
        mask = distance >= min_range
        if max_range > 0:
            mask &= distance <= max_range
        if not mask.any():
            raise ValueError(f"{name}: empty input inside the configured range")
        result[f"{name}_mask"] = mask
        result[f"{name}_xyz"] = points[mask]
    return result


def predict_frame(model, frame, device):
    # Lazy imports keep saved-result viewing independent of Torch/MinkowskiEngine.
    import torch

    from chamelion.utils.utils_dyn import MAP_TIMESTAMP, SCAN_TIMESTAMP

    with torch.inference_mode():
        scan = torch.as_tensor(frame["scan_xyz"], dtype=torch.float32, device=device)
        prior = torch.as_tensor(frame["map_xyz"], dtype=torch.float32, device=device)
        outputs = model.predict(
            scan,
            prior,
            SCAN_TIMESTAMP * torch.ones(len(scan), device=device),
            MAP_TIMESTAMP * torch.ones(len(prior), device=device),
        )
        result = dict(frame)
        for name, logits, confidence in (
            ("scan", outputs[0], outputs[2]),
            ("map", outputs[1], outputs[3]),
        ):
            if logits.shape != (len(frame[f"{name}_xyz"]),):
                raise ValueError(f"{name}: model output size mismatch")
            if confidence.shape != logits.shape:
                raise ValueError(f"{name}: confidence size mismatch")
            if not torch.isfinite(logits).all() or not torch.isfinite(confidence).all():
                raise ValueError(f"{name}: non-finite model output")
            result[f"{name}_pred"] = (logits > 0).cpu().numpy()
            result[f"{name}_logits"] = logits.cpu().numpy()
            result[f"{name}_confidence"] = confidence.cpu().numpy()
    return result
