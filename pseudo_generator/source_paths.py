"""Resolve raw input clouds without modifying existing source datasets."""
import warnings
from pathlib import Path


def source_clouds(sequence_dir):
    sequence_dir = Path(sequence_dir)
    clouds = sequence_dir / "clouds"
    if clouds.is_dir():
        return clouds
    legacy = sequence_dir / "velodyne"
    if legacy.is_dir():
        warnings.warn(
            "Using legacy velodyne/ input; new datasets should use clouds/.",
            UserWarning,
            stacklevel=2,
        )
        return legacy
    raise FileNotFoundError(f"Expected raw point clouds in {clouds}")
