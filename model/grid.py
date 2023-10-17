import torch
import torch.nn as nn
from .coordinate_utils import convert_coordinate_start


class Grid(nn.Module):
    """
    Base class for Feature level grids
    """
    def __init__(self, channels, h, w, quantization):
        super().__init__()
        self.h = h
        self.w = w
        self.channels = channels
        self.n_quant_bins = 2**quantization
        self.quant_left = (-self.n_quant_bins-1) / (2*self.n_quant_bins)
        self.quant_right = (-self.n_quant_bins) / (2*self.n_quant_bins)

        self.grid = nn.Parameter(torch.zeros(1, channels, h, w))
        nn.init.xavier_normal_(self.grid)

    def resample(self, scale_factor, coordinate_start, h, w, support_resolution_h, support_resolution_w):
        raise NotImplementedError

    def simulate_quantization(self):
        uniform = (torch.rand_like(self.grid) / self.n_quant_bins) - 1/(2*self.n_quant_bins)
        return self.grid + uniform

    @torch.no_grad()
    def clamp_values(self):
        self.grid.data.clamp_(-self.quant_left, self.quant_right)


class Grid1(Grid):
    """
    Grid1 is the high resolution grid corresponding to a specific feature level
    """
    def __init__(self, channels, h, w, quantization, circular=False):
        super().__init__(channels, h, w, quantization)
        self.circular = circular

    def resample(self, coordinate_start, h, w, support_resolution_h, support_resolution_w):
        offset_h = (support_resolution_h-1) / 2
        offset_w = (support_resolution_w-1) / 2
        if self.circular:
            scale_factor_h = int(h / self.h)
            scale_factor_w = int(w / self.w)
            assert scale_factor_h == scale_factor_w, f"Scale factors must be equal check the h: {h} and w: {w} values"
            # how to handle normalization?
            full_x, full_y = convert_coordinate_start(coordinate_start, h, w, flatten_sequence=False)
            full_x = ((full_x % self.w) - offset_w) / offset_w
            full_y = ((full_y % self.h) - offset_h) / offset_h
            full_coordinates = torch.cat((full_x, full_y), dim=-1)
            grid = torch.nn.functional.grid_sample(self.grid, full_coordinates, mode='bilinear', padding_mode='border', align_corners=True)
        else:
            full_x, full_y = convert_coordinate_start(coordinate_start, h, w, flatten_sequence=False)
            full_x = (full_x - offset_w) / offset_w
            full_y = (full_y - offset_h) / offset_h
            full_coordinates = torch.cat((full_x, full_y), dim=-1)
            grid = torch.nn.functional.grid_sample(self.grid, full_coordinates, mode='bilinear', padding_mode='border', align_corners=True)
            
        return grid
