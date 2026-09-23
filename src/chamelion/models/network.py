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

import copy

import MinkowskiEngine as ME
import torch
from pytorch_lightning import LightningModule

from chamelion.models.backbone import ConfidenceMinkUNet
from chamelion.utils.utils_dyn import *


class ChamelionNet(LightningModule):
    def __init__(self, voxel_size: float):
        super().__init__()
        self.voxel_size = voxel_size
        self.MinkUNet = ConfidenceMinkUNet(in_channels=1, out_channels=1, D=4)
        self.sigmoid = torch.nn.Sigmoid()

    def predict(self, scan_input, map_input, scan_indices, map_indices):
        def extend(tensor, batch_idx, time_idx):
            ones = torch.ones(len(tensor), 1).type_as(tensor)
            return torch.hstack([batch_idx * ones, tensor, time_idx * ones])

        # Extend to [batch_idx, x,y,z,t]
        scan_input = extend(scan_input, 0, 0)
        map_input = extend(map_input, 0, -1)

        coordinates = torch.vstack([scan_input.reshape(-1, 5), map_input.reshape(-1, 5)])
        indices = torch.vstack([scan_indices.reshape(-1, 1), map_indices.reshape(-1, 1)])

        logits, confidence = self.forward(coordinates, indices)

        # Get logits from current scan
        mask_scan = coordinates[:, 4] == 0.0
        logits_scan = logits[mask_scan]
        logits_map = logits[~mask_scan]

        confidence_scan = confidence[mask_scan]
        confidence_map = confidence[~mask_scan]

        return logits_scan, logits_map, confidence_scan, confidence_map

    def forward(self, coordinates: torch.Tensor, indices: torch.Tensor):
        quantization = torch.Tensor(
            [1.0, self.voxel_size, self.voxel_size, self.voxel_size, 1.0]
        ).type_as(coordinates)
        coordinates = torch.div(coordinates, quantization)

        # Normalize indices
        i_max = torch.max(indices)
        i_min = torch.min(indices)
        if i_min == i_max:
            features = 1.0 * torch.ones_like(indices)
        else:
            features = 1 + (i_max - indices) / (i_max - i_min)

        tensor_field = ME.TensorField(
            features=features.reshape(-1, 1), coordinates=coordinates.reshape(-1, 5)
        )

        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor, predicted_sparse_conf = self.MinkUNet(sparse_tensor)

        out = predicted_sparse_tensor.slice(tensor_field)
        out_conf = predicted_sparse_conf.slice(tensor_field)

        logits = out.features.reshape(-1)
        confidence = out_conf.features.reshape(-1)

        return logits, self.sigmoid(confidence)

    @staticmethod
    def to_label(logits):
        labels = copy.deepcopy(logits)
        mask = logits > 0
        labels[mask] = 1.0
        labels[~mask] = 0.0
        return labels
