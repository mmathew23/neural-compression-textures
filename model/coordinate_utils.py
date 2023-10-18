import torch


def convert_coordinate_start(coordinate_start, h, w, flatten_sequence=True):
    """
    Util to convert a batch of coordinate starts to a batch of coordinates of size h x w
    """
    x_offset, y_offset = torch.arange(0, w, step=1, device=coordinate_start.device), torch.arange(0, h, step=1, device=coordinate_start.device)
    xx, yy = torch.meshgrid(x_offset, y_offset)
    xx = xx.view(h, w, 1)
    yy = yy.view(h, w, 1)

    b = coordinate_start.shape[0]
    x_start, y_start = torch.split(coordinate_start, 1, dim=-1)
    # view as b x seq_len x 1
    x_start = x_start.view(b, 1, 1, 1)
    y_start = y_start.view(b, 1, 1, 1)

    full_x = x_start + xx
    full_y = y_start + yy
    if flatten_sequence:
        full_x = full_x.view(b, -1, 1)
        full_y = full_y.view(b, -1, 1)

    return full_x, full_y
