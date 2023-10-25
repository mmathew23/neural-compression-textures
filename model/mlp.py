import torch
import torch.nn as nn


class HardGELU(nn.Module):
    def __init__(self):
        super(HardGELU, self).__init__()

    def forward(self, x):
        # output 0 if x < -3/2
        left_mask = (x < -1.5)
        # output x if x > 3/2
        right_mask = (x > 1.5)
        # output x/3*(x+3/2) otherwise
        middle_mask = ~(left_mask | right_mask)

        out = torch.zeros_like(x)
        out[left_mask] = 0.0
        out[right_mask] = x[right_mask]
        out[middle_mask] = x[middle_mask] / 3 * (x[middle_mask] + 1.5)
        return out


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels

        self.layers = nn.Sequential()

        self.layers.append(nn.Conv2d(in_channels, hidden_channels, 1))
        self.layers.append(HardGELU())
        for _ in range(num_layers - 2):
            self.layers.append(nn.Conv2d(hidden_channels, hidden_channels, 1))
            self.layers.append(HardGELU())
        self.layers.append(nn.Conv2d(hidden_channels, out_channels, 1))

    def forward(self, x):
        return self.layers(x)
