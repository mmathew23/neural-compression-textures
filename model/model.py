import torch
import torch.nn as nn
from model.feature import Features
from model.mlp import MLP
from omegaconf import DictConfig


class Model(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.mlp = MLP(config.in_channels, config.hidden_channels, config.num_layers, config.out_channels)
        self.features = Features(config.features)

    def forward(self, coordinates, h, w, lod):
        features = self.features(coordinates, h, w, lod)
        return self.mlp(features)
