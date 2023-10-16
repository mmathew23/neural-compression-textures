import torch
import torch.nn as nn


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

    def resample(self, h, w):
        raise NotImplementedError

    def simulate_quantization(self):
        uniform = (torch.rand_like(self.grid) / self.n_quant_bins) - 1/(2*self.n_quant_bins)
        return self.grid + uniform

    @torch.no_grad()
    def clamp_values(self):
        self.grid.data.clamp_(-self.quant_left, self.quant_right)


class Grid0(Grid):
    """
    Grid0 is the high resolution grid corresponding to a specific feature level
    """
