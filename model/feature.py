import torch
import torch.nn as nn
from model.grid import Grid0, Grid1
from model.positional_encoding import TriangularPositionalEncoding2D
from omegaconf import DictConfig
import math


class FeatureLevel(nn.Module):
    def __init__(self, channels, h, w, quantization, lod, circular=False):
        super().__init__()
        
        self.grid0 = Grid0(channels[0], h[0], w[0], quantization, circular)
        self.grid1 = Grid1(channels[1], h[1], w[1], quantization, circular)

        self.lod = lod

    def quantize_grid_and_freeze(self):
        self.grid0.quantize_grid_and_freeze()
        self.grid1.quantize_grid_and_freeze()

    def clamp_values(self):
        self.grid0.clamp_values()
        self.grid1.clamp_values()

    def forward(self, coordinate_start, h, w, stride, resolution, quantize=False):
        return torch.cat([
            self.grid0(coordinate_start, h, w, stride, resolution, resolution, quantize=quantize),
            self.grid1(coordinate_start, h, w, stride, resolution, resolution, quantize=quantize),],
            dim=1
        )


class Features(nn.Module):
    def __init__(self, feature_config: DictConfig):
        super().__init__()
        self.resolution = feature_config.resolution
        self.circular = feature_config.circular
        self.feature_levels = nn.ModuleDict()

        self.lod_levels = int(math.log2(self.resolution))
        self.lod_offset = self.lod_levels / 2
        self.lod_scale = 2 / self.lod_offset
        self.lod_to_resolution = {i: self.resolution // (2 ** i) for i in range(self.lod_levels+1)}
        lod_check = [False for i in range(self.lod_levels+1)]
        self.feature_levels = nn.ModuleList()
        self.lod_to_feature_level = {}
        self.resolution_to_stride = {}
        for feature_idx, level in enumerate(feature_config.feature_levels):
            feature_module = FeatureLevel(
                channels=(level.grid0.channels, level.grid1.channels),
                h=(level.grid0.h, level.grid1.h),
                w=(level.grid0.w, level.grid1.w),
                quantization=level.quantization,
                circular=self.circular,
                lod=level.lod,
            )
            self.feature_levels.append(feature_module)
            # Can remove this block
            for lod_idx, lod in enumerate(level.lod):
                lod_check[lod] = True
                self.lod_to_feature_level[lod] = feature_idx
                resolution = self.lod_to_resolution[lod]
                self.resolution_to_stride[resolution] = int(2**(lod_idx+(0 if feature_idx == 0 else 1)))
            ##
        assert all(lod_check), "all lod levels must be present in config for resolution {self.resolution}"

        self.positional_encoding = TriangularPositionalEncoding2D(**feature_config.positional_encoding)

    def quantize_grid_and_freeze(self):
        for feature_level in self.feature_levels:
            feature_level.quantize_grid_and_freeze()

    def clamp_values(self):
        for feature_level in self.feature_levels:
            feature_level.clamp_values()

    def forward(self, coordinate_start, h, w, lod, quantize=False):
        if lod in self.lod_to_feature_level:
            feature_module_idx = self.lod_to_feature_level[lod]
            feature_module = self.feature_levels[feature_module_idx]
            resolution = self.lod_to_resolution[lod]
            stride = self.resolution_to_stride[resolution]
            grid_features = feature_module(coordinate_start, h, w, 1, resolution, quantize=quantize)
            # lod_features = torch.ones_like(grid_features) * (lod-self.lod_offset) / self.lod_scale
            lod_features = torch.ones(grid_features.shape[0], 1, grid_features.shape[2], grid_features.shape[3], device=grid_features.device) * (lod-self.lod_offset) / self.lod_scale
            position_features = self.positional_encoding(coordinate_start, h, w, 1)
            features = torch.cat([grid_features, lod_features, position_features], dim=1)
        else:
            raise ValueError(f"lod {lod} not found in feature levels")
        return features
