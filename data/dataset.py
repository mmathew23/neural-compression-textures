import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import math
import os
from data.material import Material
import numpy as np


def sample_lod(lod_list, p=0.05):
    # Generate the first random uniform number
    random_number = np.random.rand()

    if random_number <= p:
        # Pick a number uniformly from the list
        return np.random.choice(lod_list)
    else:
        # Generate another uniform random number
        x = np.random.rand()
        # Calculate floor(-log_4(x))
        result = np.floor(-np.log(x) / np.log(4))
        return result


def sample_tile_coordinates(image_size, tile_size, num_samples):
    # Maximum valid start coordinate
    max_start = image_size - tile_size
    # Randomly sample x and y coordinates within the valid range
    x = torch.randint(0, max_start + 1, (num_samples,))
    y = torch.randint(0, max_start + 1, (num_samples,))
    # Stack the coordinates into a list of (y, x) pairs since we like height, width order
    return torch.stack([y, x], dim=1)


class VariableTileDataset(Dataset):
    def __init__(self, image_path, tile_size, resolution, dtype, train_len):
        self.dtype = dtype
        self.resolution = resolution
        self.levels = [(i, int(self.resolution / (2**i))) for i in range(int(math.log2(self.resolution))+1)]
        self.lod_list = [i for i, _ in self.levels]
        self.image = Material()
        self.image.process_images(image_path, resolution, dtype)
        self.tile_size = tile_size
        self.channels, self.image_height, self.image_width = self.image.result_tensor.shape

        # TODO: make a more robust sampling strategy for tiles
        # Calculate how many tiles fit in the x and y directions
        self.tiles_x = self.image_width // tile_size
        self.tiles_y = self.image_height // tile_size
        self.train_len = train_len

    def __len__(self):
        return self.train_len

    def __getitem__(self, idx):
        # Extract the tile and resize to the desired resolution
        return_dict = {}
        for level, level_resolution in self.levels:
            # tile = torch.nn.functional.interpolate(self.image.result_tensor[None], size=(level_resolution, level_resolution), mode='bilinear', align_corners=True)
            tile = self.image.get_lod_resolution(level, level_resolution)
            tile_size = max(1, self.tile_size//(2**level))
            coordinates = sample_tile_coordinates(level_resolution, tile_size, 1)
            tile_y, tile_x = coordinates[0]
            tile = tile[:, tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
            return_dict[level] = {
                "pixel_values": tile,
                'resolution': level_resolution,
                'coordinates': coordinates,
                'tile_size': tile_size,
            }

        return return_dict
