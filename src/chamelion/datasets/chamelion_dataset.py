# Chamelion modifications are licensed under GPL-3.0-or-later.
# The original MapMOS copyright and MIT license notice follow.

# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
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

import hashlib
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from chamelion.config import ChamelionConfig
from chamelion.datasets import dataset_factory, sequence_dataloaders
from chamelion.utils.cache import get_cache, memoize
from chamelion.utils.utils_dyn import *

# Separate sources and protocols to prevent stale/cross-dataset cache reuse.
DATA_CACHE_VERSION = "canonical-map-no-visibility-v2"


def collate_fn(batch):
    # Returns tensor of [batch, x, y, z, t, scan_index, label]
    tensor_batch = None
    for i, (
        scan_points,
        map_points,
        scan_timestamps,
        map_timestamps,
        scan_labels,
        map_labels,
        scan_confidence_labels,
        map_confidence_labels,
    ) in enumerate(batch):
        ones = torch.ones(len(scan_points), 1).type_as(scan_points)
        scan_points = torch.hstack(
            [
                i * ones,
                scan_points,
                0.0 * ones,
                scan_timestamps,
                scan_labels,
                scan_confidence_labels,
            ]
        )

        ones = torch.ones(len(map_points), 1).type_as(map_points)
        map_points = torch.hstack(
            [
                i * ones,
                map_points,
                -1.0 * ones,
                map_timestamps,
                map_labels,
                map_confidence_labels,
            ]
        )

        tensor = torch.vstack([scan_points, map_points])
        tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])
    return tensor_batch


class ChamelionDataModule(LightningDataModule):
    """Training and validation set for Pytorch Lightning"""

    def __init__(self, dataloader: str, data_dir: Path, config: ChamelionConfig, cache_dir: Path):
        super(ChamelionDataModule, self).__init__()
        self.dataloader = dataloader
        self.data_dir = data_dir
        self.config = config
        self.cache_dir = cache_dir
        if self.cache_dir == None:
            print("No cache specified, therefore disabling shuffle during training!")
        self.shuffle = True if self.cache_dir is not None else False

        assert dataloader in sequence_dataloaders()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_set = ChamelionDataset(
                self.dataloader,
                self.data_dir,
                self.config,
                self.config.training.train,
                self.cache_dir,
            )
            self.train_loader = DataLoader(
                dataset=train_set,
                batch_size=self.config.training.batch_size,
                collate_fn=collate_fn,
                shuffle=self.shuffle,
                num_workers=self.config.training.num_workers,
                pin_memory=True,
                persistent_workers=self.config.training.num_workers > 0,
                drop_last=False,
                timeout=0,
            )
            print(f"Loaded {len(train_set):d} training samples.")

        if stage in (None, "fit", "validate"):
            val_set = ChamelionDataset(
                self.dataloader,
                self.data_dir,
                self.config,
                self.config.training.val,
                self.cache_dir,
            )
            self.valid_loader = DataLoader(
                dataset=val_set,
                batch_size=1,
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=self.config.training.num_workers,
                pin_memory=True,
                persistent_workers=self.config.training.num_workers > 0,
                drop_last=False,
                timeout=0,
            )
            print(f"Loaded {len(val_set):d} validation samples.")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class ChamelionDataset(Dataset):
    """Caches and returns scan and local maps for multiple sequences"""

    def __init__(
        self,
        dataloader: str,
        data_dir: Path,
        config: ChamelionConfig,
        sequences: list,
        cache_dir: Path,
    ):
        self.config = config
        self.sequences = sequences
        self._print = False

        # Cache
        if cache_dir is not None:
            source_key = hashlib.sha256(str(Path(data_dir).resolve()).encode()).hexdigest()[:16]
            directory = Path(cache_dir) / dataloader / DATA_CACHE_VERSION / source_key
            self.use_cache = True
            self.cache = get_cache(directory=directory)
            print("Using cache at ", directory)
        else:
            self.use_cache = False
            self.cache = None

        # Create datasets and map a sample index to the sequence and scan index
        self.datasets = {}
        self.idx_mapper = {}
        idx = 0
        for sequence in self.sequences:
            self.datasets[sequence] = dataset_factory(
                dataloader=dataloader,
                data_dir=data_dir,
                sequence=sequence,
                mode_test=False,
            )
            for sample_idx in range(len(self.datasets[sequence])):
                self.idx_mapper[idx] = (sequence, sample_idx)
                idx += 1

        self.sequence = None
        # self.odometry = Odometry(self.config.data, self.config.odometry)

    def __len__(self):
        return len(self.idx_mapper.keys())

    def __getitem__(self, idx):
        sequence, scan_index = self.idx_mapper[idx]
        (
            scan_points,
            map_points,
            scan_timestamps,
            map_timestamps,
            scan_labels,
            map_labels,
            scan_confidence_labels,
            map_confidence_labels,
        ) = self.get_scan_and_map(
            sequence,
            scan_index,
            dict(self.config.data),
        )
        return (
            scan_points,
            map_points,
            scan_timestamps,
            map_timestamps,
            scan_labels,
            map_labels,
            scan_confidence_labels,
            map_confidence_labels,
        )

    @memoize()
    def get_scan_and_map(
        self,
        sequence: int,
        scan_index: int,
        data_config_dict: Dict,
    ):
        """Returns scan points, map points in local frame and labels. Scan and map need to be in
        local frame to allow for efficient cropping (sample point does not change).
        """
        if not self._print:
            print("*****Caching now*****")
            self._print = True

        scan_points, scan_labels, timestamps, gt_pose = self.datasets[sequence][scan_index]

        # Only consider valid points
        valid_mask = scan_labels != -1
        scan_points = scan_points[valid_mask]
        scan_labels = scan_labels[valid_mask]

        if self.sequence != sequence or len(scan_points) == 0:
            self.sequence = sequence

        registered_map_points, map_labels = self.get_map_points(sequence)
        map_timestamps = MAP_TIMESTAMP * torch.ones(len(registered_map_points))

        # map_points = self.odometry.transform(
        #     registered_map_points, np.linalg.inv(gt_pose)
        # )
        map_points = local_to_global(registered_map_points, np.linalg.inv(gt_pose))

        scan_timestamps = SCAN_TIMESTAMP * np.ones(len(scan_points))

        sample_indices_scan, sample_indices_map = self.sample_static_dynamic_indices(
            scan_labels, map_labels
        )

        sampled_map_conf = self.compute_signed_distance(
            scan_points, map_points[sample_indices_map], trunc_dist=3.0
        )

        map_confidence_labels = -np.ones(len(map_points), dtype=np.float32)
        map_confidence_labels[sample_indices_map] = sampled_map_conf

        sampled_scan_conf = self.compute_signed_distance(
            map_points, scan_points[sample_indices_scan], trunc_dist=3.0
        )
        scan_confidence_labels = -np.ones(len(scan_points), dtype=np.float32)
        scan_confidence_labels[sample_indices_scan] = sampled_scan_conf

        # Keep the canonical map class targets; no hash or ray visibility mask.
        # Labels already marked -1 by the source remain ignored by the loss.

        return (
            torch.tensor(scan_points).to(torch.float32).reshape(-1, 3),
            torch.tensor(map_points).to(torch.float32).reshape(-1, 3),
            torch.tensor(scan_timestamps).to(torch.float32).reshape(-1, 1),
            torch.tensor(map_timestamps).to(torch.float32).reshape(-1, 1),
            torch.tensor(scan_labels).to(torch.float32).reshape(-1, 1),
            torch.tensor(map_labels).to(torch.float32).reshape(-1, 1),
            torch.tensor(scan_confidence_labels).to(torch.float32).reshape(-1, 1),
            torch.tensor(map_confidence_labels).to(torch.float32).reshape(-1, 1),
        )

    def get_map_points(self, sequence):
        map_points = self.datasets[sequence].gt_map_points
        gt_map_label = self.datasets[sequence].gt_map_label.copy()
        gt_map_label[gt_map_label == ND_LABEL] = PD_LABEL

        return map_points.reshape(-1, 3), gt_map_label.reshape(-1)

    def smooth_function(self, x, smooth=10, threshold=0.1, trunc_dist=0.3):
        voxel_mask = x > threshold
        trunc_mask = x >= trunc_dist
        prob = np.ones_like(x)
        prob[voxel_mask] = np.exp(-smooth * (x[voxel_mask] - threshold))
        prob[trunc_mask] = 0

        return prob

    def compute_signed_distance(
        self, scan_points, map_points, trunc_dist=0.3, smooth=10, threshold=0.1
    ):
        kdtree = cKDTree(scan_points)

        distances, indices = kdtree.query(map_points, k=1)
        truncated_distances = np.clip(distances, 0, trunc_dist)

        static_confidence = self.smooth_function(
            truncated_distances, threshold=threshold, trunc_dist=0.3, smooth=10
        )

        # return truncated_distances, static_confidence
        return static_confidence

    def sample_static_dynamic_indices(
        self, scan_labels, map_labels, scan_sample_num=2048, map_sample_num=2048
    ):
        # Static
        stat_indices_scan = np.where(scan_labels == STAT_LABEL)[0]
        stat_indices_map = np.where(map_labels == STAT_LABEL)[0]

        sample_indices_stat_scan = np.random.permutation(stat_indices_scan)[:scan_sample_num]
        sample_indices_stat_map = np.random.permutation(stat_indices_map)[:map_sample_num]

        # Negative Dynamic (ND)
        nd_indices = np.where(map_labels != STAT_LABEL)[0]
        nd_sample_num = min(map_sample_num, nd_indices.shape[0])
        sample_indices_nd = np.random.permutation(nd_indices)[:nd_sample_num]

        # Positive Dynamic (PD)
        pd_indices = np.where(scan_labels != STAT_LABEL)[0]
        pd_sample_num = min(scan_sample_num, pd_indices.shape[0])
        sample_indices_pd = np.random.permutation(pd_indices)[:pd_sample_num]

        sample_indices_map = np.concatenate([sample_indices_stat_map, sample_indices_nd])
        sample_indices_scan = np.concatenate([sample_indices_stat_scan, sample_indices_pd])

        return sample_indices_scan, sample_indices_map
