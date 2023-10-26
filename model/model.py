import torch
import torch.nn as nn
from model.feature import Features
from model.mlp import MLP
from omegaconf import DictConfig
from torchvision.utils import make_grid


class Model(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.mlp = MLP(config.in_channels, config.hidden_channels, config.num_layers, config.out_channels)
        self.features = Features(config.features)
        mean = torch.randn(config.out_channels)
        std = torch.randn(config.out_channels)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.splits = None

    def quantize_grid_and_freeze(self):
        self.features.quantize_grid_and_freeze()

    def clamp_values(self):
        self.features.clamp_values()

    def forward(self, coordinates, h, w, lod, quantize=False):
        features = self.features(coordinates, h, w, lod, quantize=quantize)
        return self.mlp(features)

    @torch.no_grad()
    def inference(self, coordinates, h, w, lod):
        assert coordinates.shape[0] == 1
        features = self.features(coordinates, h, w, lod, quantize=False)
        material = self.mlp(features)
        return self.make_grid(material[0])

    def denormalize(self, tensor):
        # This method assumes the input mean and std are tensors with the same number of elements as channels in 'tensor'
        if self.mean.shape[0] != tensor.shape[0] or self.std.shape[0] != tensor.shape[0]:
            raise ValueError("Mean and std length must be equal to the number of channels in the tensor.")

        return torch.clamp(tensor * self.std[:, None, None] + self.mean[:, None, None], 0, 1)

    def split_material(self, tensor):
        materials = torch.split(tensor, self.splits, dim=0)
        return materials

    def expand_material(self, materials, channels=3):
        h, w = materials[0].shape[-2:]

        return [mat.expand(channels, h, w) for mat in materials]

    def make_grid(self, tensor):
        materials = self.split_material(self.denormalize(tensor))
        materials = self.expand_material(materials)
        nrows = 1 if len(materials) <= 1 else 2
        #split returns a tuple but make grid expects a list
        # expand_material also converts to list
        return make_grid(materials, nrow=nrows)
