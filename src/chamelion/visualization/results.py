#!/usr/bin/env python3
"""View saved frame_*.npz evaluation results; no model or CUDA required."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Result directory or a single frame NPZ")
    args = parser.parse_args()
    from chamelion.visualization.data import SavedFrames
    from chamelion.visualization.polyscope import ChamelionViewer

    ChamelionViewer(SavedFrames(args.results)).show()


if __name__ == "__main__":
    main()
