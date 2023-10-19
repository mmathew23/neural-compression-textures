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

    def forward(self, coordinate_start, h, w, resolution):
        return torch.cat([
            self.grid0(coordinate_start, h, w, resolution, resolution),
            self.grid1(coordinate_start, h, w, resolution, resolution),],
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
            for lod in level.lod:
                lod_check[lod] = True
                self.lod_to_feature_level[lod] = feature_idx
        assert all(lod_check), "all lod levels must be present in config for resolution {self.resolution}"

        self.positional_encoding = TriangularPositionalEncoding2D(**feature_config.positional_encoding)

    def forward(self, coordinate_start, h, w, lod):
        if lod in self.lod_to_feature_level:
            feature_module_idx = self.lod_to_feature_level[lod]
            feature_module = self.feature_levels[feature_module_idx]
            resolution = self.lod_to_resolution[lod]
            grid_features = feature_module(coordinate_start, h, w, resolution)
            lod_features = torch.ones_like(grid_features) * (lod-self.lod_offset) / self.lod_scale
            position_features = self.positional_encoding(coordinate_start, h, w)
            features = torch.cat([grid_features, lod_features, position_features], dim=1)
        else:
            raise ValueError(f"lod {lod} not found in feature levels")
        return features
