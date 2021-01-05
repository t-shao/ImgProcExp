import numpy as np
from . import Image


def equalize(image):
    if image.mode is not Image.COLOR_L:
        raise NotImplementedError
    h, w = image.size
    hist = image.histogram()

    new = image.copy()
    im = new.im

    # calculate cdf
    hist_cumsum = np.cumsum(hist)
    const_a = 256 / (h * w)
    hists_cdf = (const_a * hist_cumsum).astype(np.uint8)

    # mapping
    new.im = hists_cdf[im]

    return new
