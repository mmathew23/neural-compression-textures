import torch
import torch.nn as nn
from .coordinate_utils import convert_coordinate_start


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
            div = 2**(octave+1)
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
