import math


def get_lod_from_resolution(resolution):
    return list(range(int(math.log2(resolution))+1))


def numel(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
