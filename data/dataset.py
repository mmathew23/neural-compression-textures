import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import math
import os
from data.material import Material


class VariableTileDataset(Dataset):
    def __init__(self, image_path, tile_size, resolution, dtype, train_len):
        self.dtype = dtype
        self.resolution = resolution
        self.levels = [(i, int(self.resolution / (2**i))) for i in range(int(math.log2(self.resolution))+1)]
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
            tile = torch.nn.functional.interpolate(self.image.result_tensor[None], size=(level_resolution, level_resolution), mode='bilinear', align_corners=True)
            # Calculate the top-left pixel of this tile
            tile_x = (idx % self.tiles_x) * int(self.tile_size * (level_resolution / self.image_width))
            tile_y = ((idx // self.tiles_y) % self.tiles_y) * int(self.tile_size * (level_resolution / self.image_width))

            tile_size = max(1, self.tile_size//(2**level))
            # 0 since we added an index for interpolate to work. then collation will add the index back
            tile = tile[0, :, tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
            return_dict[level] = {
                "pixel_values": tile,
                'resolution': level_resolution,
                'coordinates': torch.tensor([tile_y, tile_x], dtype=torch.long),
                'tile_size': tile_size,
            }

        return return_dict
