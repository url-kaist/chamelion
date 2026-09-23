# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path
from typing import Dict, List


def supported_file_extensions():
    return [
        "bin",
        "pcd",
        "ply",
        "xyz",
        "obj",
        "ctm",
        "off",
        "stl",
    ]


def sequence_dataloaders():
    return ["cham"]


def available_dataloaders() -> List:
    return list(dataloader_types())


def jumpable_dataloaders():
    return available_dataloaders()


def dataloader_types() -> Dict:
    # Only raw loaders belong here, not the Lightning data module or experiments.
    return {
        "cham": "ChamDataLoader",
    }


def dataset_factory(dataloader: str, data_dir: Path, *args, **kwargs):
    import importlib

    registry = dataloader_types()
    if dataloader not in registry:
        raise ValueError(f"Unsupported dataloader {dataloader!r}; choose from {list(registry)}")
    dataloader_type = registry[dataloader]
    module = importlib.import_module(f".{dataloader}", __name__)
    assert hasattr(module, dataloader_type), f"{dataloader_type} is not defined in {module}"
    dataset = getattr(module, dataloader_type)
    return dataset(data_dir=data_dir, *args, **kwargs)
