"""Verify two generation runs and their recorded source/dataset hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verify(first, second):
    first, second = Path(first), Path(second)
    a = json.loads((first / "manifest.json").read_text())
    b = json.loads((second / "manifest.json").read_text())
    for key in (
        "travel_commit",
        "config",
        "inputs_sha256",
        "dataset_sha256",
        "segmentation",
        "selected_object",
        "validation",
    ):
        if a[key] != b[key]:
            raise ValueError(f"Runs differ: {key}")
    for directory, manifest in ((first, a), (second, b)):
        if hashes(directory / "Const_pseudo_dataset") != manifest["dataset_sha256"]:
            raise ValueError(f"Dataset does not match manifest: {directory}")
        if not manifest["source_inputs_unchanged"] or manifest["training_executed"]:
            raise ValueError(f"Unexpected source/training status: {directory}")
    if hashes(first / "preparation") != hashes(second / "preparation"):
        raise ValueError("TRAVEL segmentation or prepared maps differ")
    return {
        "identical_dataset_files": len(a["dataset_sha256"]),
        "identical_preparation_files": len(hashes(first / "preparation")),
        "travel_commit": a["travel_commit"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.first, args.second), indent=2))
