import numpy as np
import cmath
from . import Image


# TODO: to be tested further
def dft2(image):
    if not isinstance(image, Image.Image):
        raise TypeError
    if image.mode is not Image.COLOR_L:
        raise ValueError
    im = image.im
    h, w = image.size

    im_freq = np.zeros((h, w), dtype=complex)
    for k in range(h):
        for l in range(w):
            sum_matrix = 0.0
            for m in range(h):
                for n in range(w):
                    e = cmath.exp(- 2j * np.pi * ((k * m) / h + (l * n) / w))
                    sum_matrix += im[m, n] * e
            im_freq[k, l] = sum_matrix

    return im_freq

# TODO: idft2

# TODO: fft2 & ifft2

# TODO: dct2 & idct2

# TODO: dwt & idwt
