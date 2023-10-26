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
        self.quant_left = -(self.n_quant_bins-1) / (2*self.n_quant_bins)
        self.quant_right = (self.n_quant_bins) / (2*self.n_quant_bins)
        self.circular = circular

        self.grid = nn.Parameter(torch.randn(1, channels, h, w)/100)

    def resample(self, coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=False):
        raise NotImplementedError

    def simulate_quantization(self):
        uniform = (torch.rand_like(self.grid) / self.n_quant_bins) - 1/(2*self.n_quant_bins)
        return self.grid + uniform

    @torch.no_grad()
    def quantize_grid_and_freeze(self):
        bin_edges = torch.linspace(self.quant_left, self.quant_right, steps=self.n_quant_bins+1, device=self.grid.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_indices = torch.bucketize(self.grid, bin_edges, right=True) - 1
        bin_indices.clamp_(0, self.n_quant_bins - 1) # ensure within range

        self.grid.data = bin_centers[bin_indices]
        self.grid.requires_grad_(False)

    @torch.no_grad()
    def clamp_values(self):
        self.grid.data.clamp_(self.quant_left, self.quant_right)

    def forward(self, coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=False):
        return self.resample(coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=quantize)


class Grid0(Grid):
    """
    Grid0 is the high resolution grid corresponding to a specific feature level
    """
    def __init__(self, channels, h, w, quantization, circular=False):
        super().__init__(channels, h, w, quantization, circular)

    def resample(self, coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=False):
        # Support resolution not needed since normalization is determined by main grid size
        scale_factor_h = support_resolution_h / self.h
        scale_factor_w = support_resolution_w / self.w
        assert scale_factor_h == scale_factor_w, f"Scale factors must be equal check the h: {h} and w: {w} values"

        # meshgrid returns the opposite convention of what grid_sample uses
        full_x, full_y = convert_coordinate_start(coordinate_start, h, w, stride=stride, flatten_sequence=False)

        # we don't want to sample in between points so we need to round
        full_x = (full_x // scale_factor_w).long()
        full_y = (full_y // scale_factor_h).long()
        grids = []
        if quantize:
            grid = self.simulate_quantization()
        else:
            grid = self.grid

        _, c, h1, w1 = grid.shape
        b = full_x.shape[0]
        grid = grid.expand(b, c, h1, w1)
        for offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if self.circular:
                full_x_offset = (full_x+offset[0]) % self.w
                full_y_offset = (full_y+offset[1]) % self.h
            else:
                full_x_offset = torch.clamp(full_x+offset[0], 0, self.w-1)
                full_y_offset = torch.clamp(full_y+offset[1], 0, self.h-1)
            grids.append(grid.view(b, c, -1).gather(2, (full_y_offset * w1 + full_x_offset).view(b, 1, -1).expand(b, c, -1)).view(b, c, h, w))

        return torch.cat(grids, dim=1)


class Grid1(Grid):
    """
    Grid1 is the low resolution grid corresponding to a specific feature level
    """
    def __init__(self, channels, h, w, quantization, circular=False):
        super().__init__(channels, h, w, quantization, circular)

    def resample(self, coordinate_start, h, w, stride, support_resolution_h, support_resolution_w, quantize=False):
        # support resolution does the scaling so that it samples the grid appropriately
        scale_factor_h = support_resolution_h / self.h
        scale_factor_w = support_resolution_w / self.w
        assert scale_factor_h == scale_factor_w, f"Scale factors must be equal check the h: {h} and w: {w} values"

        offset_h = (support_resolution_h-1) / 2
        offset_w = (support_resolution_w-1) / 2
        # meshgrid returns the opposite convention of what grid_sample uses
        full_x, full_y = convert_coordinate_start(coordinate_start, h, w, stride=stride, flatten_sequence=False)
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

        if quantize:
            grid = self.simulate_quantization()
        else:
            grid = self.grid
        grid = torch.nn.functional.grid_sample(grid.expand([full_coordinates.shape[0], self.grid.shape[1], self.grid.shape[2], self.grid.shape[3]]), full_coordinates, mode='bilinear', padding_mode='border', align_corners=True)

        return grid
