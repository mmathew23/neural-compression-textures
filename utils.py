import math


def get_lod_from_resolution(resolution):
    return list(range(int(math.log2(resolution))+1))