import torch
import torch.nn as nn
from model.coordinate_utils import convert_coordinate_start

def batched_gather(grid):
    # Assuming 'grid' is your original grid tensor and 'coords' is your bxhxwx2 tensor of coordinates.
    # 'grid' is of shape: [C, H, W] (channels, height, width)
    # 'coords' is of shape: [B, H', W', 2] (batch size, height, width, 2 for x and y coordinates)

    # Normalize coordinates from -1 to 1 (if they're not already)
    coords = coords * 2 - 1

    # Convert these coordinates to match the grid size, assuming they are in range [-1, 1]
    coords[..., 0] = (coords[..., 0] + 1) * (grid.size(2) - 1) / 2
    coords[..., 1] = (coords[..., 1] + 1) * (grid.size(1) - 1) / 2

    # Separate x and y coordinates and round them to make them integers
    x_coords = torch.round(coords[..., 0]).long()
    y_coords = torch.round(coords[..., 1]).long()

    # Create a range tensor for the channels, and view it to make it broadcastable with the coordinates
    c_range = torch.arange(grid.size(0), device=grid.device).view(1, -1, 1, 1)

    # Use advanced indexing to gather the data from the grid
    output = grid[:, y_coords, x_coords]

    # We need to add the channel dimension back, and transpose the tensor to the right shape
    output = output.unsqueeze(1)
    output = output.expand(-1, grid.size(0), -1, -1)  # Expand channel dimension
    output = c_range * output  # This utilizes broadcasting to match channel indices

    # Now 'output' is a tensor of shape [B, C, H', W'] containing the sampled values from the grid

def tri(x, offset=0.5):
    """
    Compute the triangular wave for a tensor of inputs.

    Arguments:
    x : torch.Tensor, the input tensor for which the triangular wave is computed.

    Returns:
    torch.Tensor, the triangular wave values for the input tensor.
    """
    return 2 * torch.abs((x-offset) % 2 - 1) - 1


# (1 freq, 0 offset), (2 freq, 0.5 offset), (2 freq, 0 offset), (4 freq, 0.5 offset), (4 freq, 0 offset)
class TriangularPositionalEncoding1D(nn.Module):
    def __init__(self, sequence_length=8, octaves=3, include_constant=True):
        super().__init__()

        self.include_constant = include_constant
        self.octaves = octaves
        self.sequence_length = sequence_length
        encodings = []
        # I am basing the positional encoding config based on Fig 5 in the paper
        # Authors don't seem to detail the exact config, and closest thing is the
        # pixel values from the paper
        x = torch.arange(0, sequence_length, step=1)
        for octave in range(octaves):
            div = 2**(octave)
            for i, offset in enumerate((.0, 0.5)):
                if octave == 0 and i == 1:
                    # Skip the second offset in the first octave
                    continue
                encoding = tri(x / (div), offset=offset)
                encodings.append(encoding)
        if include_constant:
            encodings.append(torch.zeros(sequence_length, dtype=encodings[-1].dtype))
        encodings = torch.stack(encodings)
        self.register_buffer('encodings', encodings)

    def forward(self, coordinates):
        """
        Compute the triangular wave for coordinates

        Arguments:
        coordinates: bxseq_lenx1 tensor of coordinates

        Returns:
        torch.Tensor, the triangular wave values
        """
        b = coordinates.shape[0]
        d1, d2 = self.encodings.shape
        encodings = self.encodings.unsqueeze(0).expand(b, d1, d2)
        results = torch.gather(encodings, 2, (coordinates % self.sequence_length).permute(0, 2, 1))
        return results


class TriangularPositionalEncoding2D(nn.Module):
    def __init__(self, sequence_length=8, octaves=3, include_constant=True):
        super().__init__()

        self.include_constant = include_constant
        self.octaves = octaves
        self.sequence_length = sequence_length
        self.encoding = TriangularPositionalEncoding1D(sequence_length, octaves, include_constant)

    def forward(self, coordinates, h, w):
        """
        Compute the triangular wave for a batch of start coordinates

        Arguments:
        coordinates: bx2 tensor of start coordinates
        h: height of the image
        w: width of the image

        Returns:
        torch.Tensor, the triangular wave values
        """
        full_x, full_y = convert_coordinate_start(coordinates, h, w)
        b = coordinates.shape[0]
        encoding_x = self.encoding(full_x).view(b, h, w, -1).permute(0, 3, 1, 2)
        encoding_y = self.encoding(full_y).view(b, h, w, -1).permute(0, 3, 1, 2)
        return torch.cat([encoding_x, encoding_y], dim=1)
