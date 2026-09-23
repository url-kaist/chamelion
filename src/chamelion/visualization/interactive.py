#!/usr/bin/env python3
"""Label-free, fixed-pose raw inference with a Polyscope viewer (no temporal fusion)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trusted model checkpoint")
    parser.add_argument(
        "--map", required=True, type=Path, help="Prior map PCD in global coordinates"
    )
    parser.add_argument("--scans", required=True, type=Path, help="Directory of numeric PCD scans")
    parser.add_argument(
        "--poses", required=True, type=Path, help="Full poses.txt, 12 values per row"
    )
    parser.add_argument("--scan-coordinates", required=True, choices=["global", "local"])
    parser.add_argument("--config", type=Path, default=Path("config/cham.yaml"))
    parser.add_argument("--output", required=True, type=Path, help="New directory; never overwrite")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; existing data is preserved")
    import numpy as np
    import torch

    from chamelion.config import load_config
    from chamelion.inference.predict import PROTOCOL, predict_frame
    from chamelion.utils.checkpoint import load_model
    from chamelion.visualization.data import CloudFrames
    from chamelion.visualization.polyscope import ChamelionViewer

    cfg = load_config(args.config)
    frames = CloudFrames(
        args.map,
        args.scans,
        args.poses,
        args.scan_coordinates,
        cfg.mos.min_range_mos,
        cfg.mos.max_range_mos,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Interactive model inference requires a working CUDA environment")
    torch.set_num_threads(4)
    torch.manual_seed(66)
    np.random.seed(66)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model = load_model(args.checkpoint, cfg.mos.voxel_size_mos, device)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    def infer(frame):
        result = predict_frame(model, frame, device)
        result["protocol"] = np.asarray(PROTOCOL)
        result["checkpoint_sha256"] = np.asarray(digest.hexdigest())
        return result

    viewer = ChamelionViewer(frames, infer=infer, output=args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "run.json").open("x") as stream:
        json.dump(
            {
                "protocol": PROTOCOL,
                "checkpoint_sha256": digest.hexdigest(),
                "scan_coordinates": args.scan_coordinates,
                "config": cfg.model_dump(mode="json"),
                "threshold": "logit > 0",
                "confidence_filter": False,
                "temporal_fusion": False,
                "visibility_mask": False,
            },
            stream,
            indent=2,
        )
    viewer.show()


if __name__ == "__main__":
    main()
