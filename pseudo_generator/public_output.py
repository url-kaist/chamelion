"""Keep interactive preparation artifacts separate from the public dataset."""
import shutil
import tempfile
from pathlib import Path


def prepare_output(config):
    output = config["output_dataset"]
    name = output["sequence"]
    if not name.startswith("sequence_") or not name[9:].isdigit():
        raise ValueError("output_dataset.sequence must be sequence_<number>")
    source = config["load_dataset"]
    pose_file = Path(source["dataset_root"]) / "sequences" / source["seq"] / "poses.txt"
    if not pose_file.is_file():
        raise FileNotFoundError(pose_file)
    sequence_dir = Path(output["root"]) / "sequences" / name
    try:
        sequence_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Preserve the old sequence, including partial exports. Retry elsewhere.
        retry_root = Path(tempfile.mkdtemp(prefix="generation_", dir=Path(output["root"]).parent))
        sequence_dir = retry_root / "Const_pseudo_dataset" / "sequences" / name
        sequence_dir.mkdir(parents=True, exist_ok=False)
        print(f"Existing output preserved. New sequence: {sequence_dir}", flush=True)
    shutil.copy2(pose_file, sequence_dir / "poses.txt")
    return sequence_dir / "submaps"
