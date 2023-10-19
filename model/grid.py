import torch
import torch.nn as nn
from .coordinate_utils import convert_coordinate_start


class Grid(nn.Module):
    """
    Base class for Feature level grids
    """
    def __init__(self, channels, h, w, quantization, circular=False):
        super().__init__()
        self.h = h
        self.w = w
        self.channels = channels
        self.n_quant_bins = 2**quantization
        self.quant_left = (-self.n_quant_bins-1) / (2*self.n_quant_bins)
        self.quant_right = (-self.n_quant_bins) / (2*self.n_quant_bins)
        self.circular = circular

        self.grid = nn.Parameter(torch.zeros(1, channels, h, w))
        nn.init.xavier_normal_(self.grid)

    def resample(self, coordinate_start, h, w, support_resolution_h, support_resolution_w):
        raise NotImplementedError

    def simulate_quantization(self):
        uniform = (torch.rand_like(self.grid) / self.n_quant_bins) - 1/(2*self.n_quant_bins)
        return self.grid + uniform

    @torch.no_grad()
    def clamp_values(self):
        self.grid.data.clamp_(-self.quant_left, self.quant_right)

    def forward(self, coordinate_start, h, w, support_resolution_h, support_resolution_w):
        return self.resample(coordinate_start, h, w, support_resolution_h, support_resolution_w)


class Grid0(Grid):
    """
    Grid0 is the high resolution grid corresponding to a specific feature level
    """
    def __init__(self, channels, h, w, quantization, circular=False):
        super().__init__(channels, h, w, quantization, circular)

    def resample(self, coordinate_start, h, w, support_resolution_h=None, support_resolution_w=None):
        # Support resolution not needed since normalization is determined by main grid size
        scale_factor_h = h / self.h
        scale_factor_w = w / self.w
        assert scale_factor_h == scale_factor_w, f"Scale factors must be equal check the h: {h} and w: {w} values"

        # it's not self.h - 1 because we are returning one extra coordinate
        offset_h = self.h / 2
        offset_w = self.w / 2
        # meshgrid returns the opposite convention of what grid_sample uses
        # returns plus 1 in each direction so that we can concatenate them to a neural network
        # to upsample
        full_x, full_y = convert_coordinate_start(coordinate_start, h+1, w+1, flatten_sequence=False)

        # we don't want to sample in between points so we need to round
        full_x = full_x // scale_factor_w
        full_y = full_y // scale_factor_h
        if self.circular:
            full_x = ((full_x % self.w) - offset_w) / offset_w
            full_y = ((full_y % self.h) - offset_h) / offset_h
        else:
            # extra coordinate in this case should follow grid samples padding_mode
            full_x = (full_x - offset_w) / offset_w
            full_y = (full_y - offset_h) / offset_h
        full_coordinates = torch.cat((full_y, full_x), dim=-1)
        full_coordinates = full_coordinates.to(device=self.grid.device, dtype=self.grid.dtype)
        grid = torch.nn.functional.grid_sample(self.grid.expand([full_coordinates.shape[0], self.grid.shape[1], self.grid.shape[2], self.grid.shape[3]]), full_coordinates, mode='bilinear', padding_mode='border', align_corners=True)
        grid = torch.cat([grid[:, :, :-1, :-1], grid[:, :, 1:, :-1], grid[:, :, :-1, 1:], grid[:, :, 1:, 1:]], dim=1)
        return grid


class Grid1(Grid):
    """
    Grid1 is the low resolution grid corresponding to a specific feature level
    """
    def __init__(self, channels, h, w, quantization, circular=False):
        super().__init__(channels, h, w, quantization, circular)

    def resample(self, coordinate_start, h, w, support_resolution_h, support_resolution_w):
        # support resolution does the scaling so that it samples the grid appropriately
        scale_factor_h = h / self.h
        scale_factor_w = w / self.w
        assert scale_factor_h == scale_factor_w, f"Scale factors must be equal check the h: {h} and w: {w} values"

        offset_h = (support_resolution_h-1) / 2
        offset_w = (support_resolution_w-1) / 2
        # meshgrid returns the opposite convention of what grid_sample uses
        full_x, full_y = convert_coordinate_start(coordinate_start, h, w, flatten_sequence=False)
        if self.circular:
            # normalize and account for circular
            full_x = ((full_x % support_resolution_w) - offset_w) / offset_w
            full_y = ((full_y % support_resolution_h) - offset_h) / offset_h
        else:
            # normalize
            full_x = (full_x - offset_w) / offset_w
            full_y = (full_y - offset_h) / offset_h
        full_coordinates = torch.cat((full_y, full_x), dim=-1)
        full_coordinates = full_coordinates.to(device=self.grid.device, dtype=self.grid.dtype)
        grid = torch.nn.functional.grid_sample(self.grid.expand([full_coordinates.shape[0], self.grid.shape[1], self.grid.shape[2], self.grid.shape[3]]), full_coordinates, mode='bilinear', padding_mode='border', align_corners=True)

        return grid
