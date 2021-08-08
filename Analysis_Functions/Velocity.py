import numpy as np
from Setup.MazeFunctions import ConnectAngle
from scipy.ndimage import gaussian_filter


max_Vel_trans, max_Vel_angle = {'XS': 4, 'S': 4, 'M': 2, 'L': 2, 'SL': 2, 'XL': 2}, \
                               {'XS': 10, 'S': 10, 'M': 2, 'L': 2, 'SL': 2, 'XL': 2}


def crappy_velocity(x, i):
    return x.position[min(i + x.fps, x.position.shape[0] - 1)] - x.position[i]


