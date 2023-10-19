import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torchvision.transforms.functional import to_tensor, normalize
import math


class VariableTileDataset(Dataset):
    def __init__(self, image_path, tile_size, resolution):
        self.resolution = resolution
        self.levels = [(i, int(self.resolution / (2**i))) for i in range(int(math.log2(self.resolution))+1)]
        self.image = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.BICUBIC)
        self.tile_size = tile_size

        # Calculate how many tiles fit in the x and y directions
        self.tiles_x = self.image.width // tile_size
        self.tiles_y = self.image.height // tile_size

    def __len__(self):
        return self.image.width * self.image.height

    def __getitem__(self, idx):
        # Extract the tile and resize to the desired resolution
        return_dict = {}
        for level, level_resolution in self.levels:
            tile = self.image.resize((level_resolution, level_resolution), Image.BILINEAR)
            # Calculate the top-left pixel of this tile
            tile_x = (idx % tile.width)
            tile_y = (idx // tile.width)

            tile = tile.crop((tile_x, tile_y, tile_x + self.tile_size, tile_y + self.tile_size))
            return_dict[level] = {
                "pixel_values": normalize(to_tensor(tile), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                'resolution': level_resolution,
                'coordinates': torch.tensor([tile_y, tile_x], dtype=torch.long),
                'tile_size': self.tile_size,
            }

        return return_dict
