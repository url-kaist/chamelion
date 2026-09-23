"""Create a fresh preparation configuration while preserving existing labels."""
import copy
import tempfile
from pathlib import Path

import yaml


def create_retry_config(config, source_config, destination):
    destination = Path(destination).resolve()
    result = copy.deepcopy(config)
    name = result["output_dataset"]["sequence"]
    if not name.startswith("sequence_") or not name[9:].isdigit():
        raise ValueError("Public sequence must be sequence_<number>")
    # Preserve old label locations explicitly; never rename or regenerate them.
    source = result["load_dataset"]
    source.setdefault("pred_inst_sequence", source["seq"])
    result["travel_config"] = str(
        (Path(source_config).resolve().parent / result["travel_config"]).resolve()
    )
    # Allocate a new filename on every retry; never overwrite an old config.
    requested = destination
    attempt = 0
    while True:
        try:
            stream = destination.open("x")
            break
        except FileExistsError:
            attempt += 1
            destination = requested.with_name(f"{requested.stem}_{attempt:03d}{requested.suffix}")
    with stream:
        run_dir = Path(tempfile.mkdtemp(prefix="preparation_", dir=destination.parent))
        source["pred_save_dir"] = str(run_dir / "prepared" / name)
        result["output_dataset"]["root"] = str(run_dir / "Const_pseudo_dataset")
        yaml.safe_dump(result, stream, sort_keys=False)
    return result, destination
