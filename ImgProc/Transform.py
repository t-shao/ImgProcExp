import numpy as np
import cmath
from . import Image


def DFT2D(image):
    if not isinstance(image, Image.Image):
        raise TypeError
    if image.mode is not Image.COLOR_L:
        raise ValueError
    im = image.im
    M, N = image.size  # (img x, img y)

    dft2d = np.zeros((M, N), dtype=complex)
    for k in range(M):
        for l in range(N):
            sum_matrix = 0.0
            for m in range(M):
                for n in range(N):
                    e = cmath.exp(- 2j * np.pi * ((k * m) / M + (l * n) / N))
                    sum_matrix += im[m, n] * e
            dft2d[k, l] = sum_matrix

    return dft2d
